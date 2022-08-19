import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from torchmetrics.functional.classification import auroc
from transformers import RobertaModel, AdamW, get_linear_schedule_with_warmup, logging

categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_categories = len(categories)
model_name = 'roberta-base'

class SentenceClassifier(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        logging.set_verbosity_error()
        self.bert = RobertaModel.from_pretrained(model_name, return_dict = True)
        if self.config['pre_classifier'] is not None:
            self.pre_classifier = nn.Linear(self.bert.config.hidden_size, 768)
            nn.init.xavier_uniform_(self.pre_classifier.weight)
        else:
            self.classifier = nn.Linear(self.bert.config.hidden_size, self.config['n_classes'])
            nn.init.xavier_uniform_(self.classifier.weight)
        if self.config['dropout']:
            self.dropout = torch.nn.Dropout(0.2)
        self.classifier = nn.Linear(768, self.config['n_classes'])
        nn.init.xavier_uniform_(self.classifier.weight)
        # loss functions
        if self.config['class_weight'] is not None:
            self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1, 7, 3, 2, 1, 2, 15, 2]), reduction='none')
        else:
            print('no class weight')
            self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')


    def forward(self, input_ids, attention_mask, sample_weights=None, labels=None):
        # distillBert layer
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # distilBert output method
        if self.config['pool_method'] == 'pooler':
            pooled_output = output.pooler_output
        elif self.config['pool_method'] == 'mean':
            pooled_output = torch.mean(output.last_hidden_state,1)

        # drop out
        if self.config['dropout']:
            pooled_output = self.dropout(pooled_output)

        # tahn or relu
        if self.config['pre_classifier'] is not None:
            pooled_output = self.pre_classifier(pooled_output)
        if self.config['pre_classifier'] == 'tanh':
            pooled_output = torch.tanh(pooled_output)
        else:
            pooled_output = torch.relu(pooled_output)

        # final logits
        logits = self.classifier(pooled_output)

        # calculate loss
        loss = 0
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.config['n_classes']), labels.view(-1, self.config['n_classes']))
            if self.config['sample_weight'] and sample_weights is not None:
                loss = loss * sample_weights
            loss = torch.mean(loss)
        return loss, logits


    def training_step(self, batch, batch_index):
        loss, outputs = self(**batch)
        # self.log("train loss", loss, prog_bar = True, logger = True)
        return {"loss": loss, "predictions": outputs, "labels": batch["labels"]}

    def validation_step(self, batch, batch_index):
        loss, outputs = self(**batch)
        # self.log("validation loss", loss, prog_bar = True, logger = True)
        return {"val loss": loss, "predictions": outputs, "labels": batch["labels"]}

    def test_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        # self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_index):
        _, logits = self(**batch)
        return logits

    def training_epoch_end(self, outputs):
        labels,predictions, roc =[],[],[]

        for output in outputs:
            for output_labels in output['labels'].detach().cpu():
                labels.append(output_labels)

            for output_preds in output['predictions'].detach().cpu():
                predictions.append(output_preds)
                
        labels = torch.stack(labels).int()
        predictions = torch.sigmoid(torch.stack(predictions))

        for i, categorie in enumerate(categories):
            roc_score = auroc(predictions[:, i], labels[:, i].int())
            roc.append(roc_score)
            # self.logger.experiment.add_scalar(f"Train/{categorie}_roc_auc", roc_score, self.current_epoch)
        # self.log("ptl/train_avg_roc", np.mean(roc))

    def validation_epoch_end(self, outputs):
        labels,predictions,roc =[],[],[]
        
        for output in outputs:
            for output_labels in output['labels'].detach().cpu():
                labels.append(output_labels)
            for output_preds in output['predictions'].detach().cpu():
                predictions.append(output_preds)

        labels = torch.stack(labels)
        predictions = torch.sigmoid(torch.stack(predictions))

        for i, categorie in enumerate(categories):
            roc_score = auroc(predictions[:, i], labels[:, i].int())
            roc.append(roc_score)
            # self.logger.experiment.add_scalar(f"Validation/{categorie}_roc_auc", roc_score, self.current_epoch)
        # self.log("ptl/val_avg_roc", np.mean(roc))

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        total_steps = self.config['train_size']/self.config['batch_size']
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer],[scheduler]

