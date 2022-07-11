from multiprocessing import freeze_support
import torch
import math
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torchmetrics.functional.classification import auroc
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
from transformers import AutoModel, BertModel, AdamW, AutoTokenizer, BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_categories = len(categories)
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'

class sentencesDataset(Dataset):

    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_token_len: int = 30):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        sentence = str(row.Sentence)
        labels = row[categories]
        encoding = self.tokenizer.encode_plus(sentence, 
                                            add_special_tokens=True, 
                                            return_tensors='pt',
                                            truncation=True,
                                            max_length=self.max_token_len, 
                                            padding='max_length',
                                            return_attention_mask=True)

        return {'input_ids': encoding.input_ids.flatten(), 
                'attention_mask': encoding.attention_mask.flatten(), 
                'labels': torch.FloatTensor(labels)}

class DataModule(pl.LightningDataModule):

        def __init__(self, data_train, data_test, tokenizer, batch_size = 8, max_token_len = 128):
            super().__init__()
            self.data_train = data_train
            self.data_test = data_test
            self.batch_size = batch_size
            self.max_token_len = max_token_len
            self.tokenizer = tokenizer

        def setup(self, stage=None):
            self.train_dataset = sentencesDataset(self.data_train, self.tokenizer)
            self.test_dataset  = sentencesDataset(self.data_test, self.tokenizer)
            #self.val_dataset = sentencesDataset(self.data_val, self.tokenizer)

        def train_dataloader(self):
            return DataLoader(self.train_dataset,batch_size=self.batch_size,num_workers=2,shuffle=True)
        def val_dataloader(self):
            return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=1)
        def predict_dataloader(self):
            return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=2, shuffle=False)

class SentenceClassifier(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # distillBert Layer
        self.bert = BertModel.from_pretrained(model_name, return_dict = True)
        # classifier and pre-classifier layers -------
        if self.config['pre_classifier'] is not None:
            self.pre_classifier = torch.nn.Linear(self.bert.config.hidden_size, 64)
            self.classifier = torch.nn.Linear(64, self.config['n_classes'])
            torch.nn.init.xavier_uniform_(self.pre_classifier.weight)
            torch.nn.init.xavier_uniform_(self.classifier.weight)
        else:
            self.classifier = torch.nn.Linear(self.bert.config.hidden_size, self.config['n_classes'])
            torch.nn.init.xavier_uniform_(self.classifier.weight)
        # drop outs -------
        if self.config['dropout']:
            self.dropout = torch.nn.Dropout(0.2)
        # loss functions --------
        if self.config['class_weight'] is not None:
            self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.config['class_weight'], reduction='none')
        else:
            print('no class weight')
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')


    def forward(self, input_ids, attention_mask, sample_weights=None, labels=None):
        # distillBert layer
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # roberta output method
        if self.config['pool_method'] == 'pooler':
            pooled_output = output.pooler_output
        elif self.config['pool_method'] == 'mean':
            pooled_output = torch.mean(output.last_hidden_state,1)

        # tahn or relu
        if self.config['pre_classifier'] is not None:
            pooled_output = self.pre_classifier(pooled_output)
        if self.config['pre_classifier'] == 'tanh':
            pooled_output = torch.tanh(pooled_output)
        else:
            pooled_output = torch.relu(pooled_output)

        # drop out
        if self.config['dropout']:
            pooled_output = self.dropout(pooled_output)

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
        self.log("train loss", loss, prog_bar = True, logger = True)
        return {"loss": loss, "predictions": outputs, "labels": batch["labels"]}

    def validation_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val loss", loss, prog_bar = True, logger = True)
        return {"val_loss": loss, "predictions":outputs, "labels": labels}


    def predict_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        return labels

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
            self.logger.experiment.add_scalar(f"Train/{categorie}_roc_auc", roc_score, self.current_epoch)
        self.log("ptl/train_avg_roc", np.mean(roc))

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
            self.logger.experiment.add_scalar(f"Validation/{categorie}_roc_auc", roc_score, self.current_epoch)
        self.log("ptl/val_avg_roc", np.mean(roc))

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        total_steps = self.config['train_size']/self.config['batch_size']
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer],[scheduler]

def main(): 
    categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
    n_categories = len(categories)

    # Split dataset: train 80, test 20 + shuffle 
    def split_data(data_path):
        data = pd.read_pickle(data_path)
        data = data.drop(columns=['index'])
        x_train, x_test = train_test_split(data, test_size=0.2, shuffle=True)
        #x_val, x_test = train_test_split(x_valAndTest, test_size=0.66, shuffle=True)
        return x_train, x_test

    data_train, data_test = split_data("data/data_huggingface.pkl")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, return_dict=True)

    data_module = DataModule(data_train, data_test, tokenizer, batch_size=32)
    data_module.setup()
    
    config = {
        "lr": 1e-5,
        "batch_size": 32,
        "max_token_len": 128,
        "n_epochs": 10,
        "weight_decay":0.01,
        "n_classes":n_categories,
        "warmup":0.3,
        "dropout":True,
        "train_size": len(data_module.train_dataloader()),
        "pre_classifier":'relu',
        "pool_method":'mean',
        "class_weight":None,
        'sample_weight': None,
        "optim":'sgd'
    }

    # Modeling
    model = SentenceClassifier(config)
    trainer = pl.Trainer(max_epochs=config['n_epochs'], num_sanity_val_steps=50)
    checkpoint_path = 'output/preTrainedModel.ckpt'
    trainer.fit(model, data_module,ckpt_path=checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)

    # true labels 
    true_labels = np.array(data_test[categories])

    # pred labels 
    def predict(m, dm, ckpt_path):
        predictions = trainer.predict(m, dm, ckpt_path)
        flattened_predictions = [p>0.5 for batch in predictions for p in batch]
        flattened_predictions = np.stack(flattened_predictions)
        return flattened_predictions
    
    pred_labels = predict(model, data_module, checkpoint_path)

    # Converting to binary
    to_binary = {True: 1, False: 0}
    pred_labels = np.vectorize(to_binary.get)(pred_labels)
    true_labels = np.vectorize(to_binary.get)(true_labels)

    fpr = dict()
    tpr = dict()
    auc = dict()
    for i in range(n_categories):
        fpr[i], tpr[i], _ = metrics.roc_curve(true_labels[:,i], pred_labels[:,i])
        auc[i] = metrics.roc_auc_score(true_labels[:,i], pred_labels[:, i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{categories[i]}, {auc[i]}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="best")
    plt.title('ROC curve') 
    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()