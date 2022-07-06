from multiprocessing import freeze_support
from sqlite3 import Row
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import auroc
import math
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
# from transformers import DataCollatorWithPadding
# import tensorflow as tf
# from tensorflow.data import Dataset 
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.schedules import PolynomialDecay
# from keras.losses import BinaryCrossentropy
# from keras.metrics import AUC, accuracy
import pandas as pd 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def main(): 
    categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
    n_categories = len(categories)

    # Split dataset: train 70, val 10, test 20 + shuffle 
    def split_data(data_path):
        data = pd.read_pickle(data_path)
        data = data.drop(columns=['index'])
        x_train, x_valAndTest = train_test_split(data, test_size=0.3, shuffle=True)
        x_val, x_test = train_test_split(x_valAndTest, test_size=0.66, shuffle=True)
        return x_train, x_val, x_test

    data_train, data_val, data_test = split_data("data\data_huggingface.pkl")

    model_name = "distilbert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class sentencesDataset(Dataset):

        def __init__(self, data: pd.DataFrame, tokenizer: tokenizer, max_token_len: int = 30):
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

        def __init__(self, data_train, data_test, data_val, tokenizer, batch_size = 16, max_token_len = 30):
            super().__init__()
            self.data_train = data_train
            self.data_test = data_test
            self.data_val = data_val
            self.batch_size = batch_size
            self.max_token_len = max_token_len
            self.tokenizer = tokenizer

        def setup(self, stage=None):
            self.train_dataset = sentencesDataset(self.data_train, self.tokenizer)
            self.test_dataset  = sentencesDataset(self.data_test, self.tokenizer)
            self.val_dataset = sentencesDataset(self.data_val, self.tokenizer)

        def train_dataloader(self):
            return DataLoader(self.train_dataset,batch_size=self.batch_size,num_workers=4,shuffle=True)
        def val_dataloader(self):
            return DataLoader(self.val_dataset,batch_size=self.batch_size,num_workers=4)
        def test_dataloader(self):
            return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=4, shuffle=False)

    class SentenceClassifier(pl.LightningModule):

        def __init__(self, config: dict):
            super().__init__()
            self.config = config
            self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict=True)
            self.hidden = nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
            self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])
            torch.nn.init.xavier_uniform_(self.hidden.weight)
            torch.nn.init.xavier_uniform_(self.classifier.weight)
            self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
            self.dropout = nn.Dropout() # prevents/reduces overfitting


        def forward(self, input_ids, attention_mask, labels=None):
            # distilled BERT model
            output = self.pretrained_model(input_ids = input_ids, attention_mask = attention_mask)
            pooled_output = torch.mean(output.last_hidden_state, 1)
            # nn classification layers
            pooled_output = self.hidden(pooled_output)
            pooled_output = self.dropout(pooled_output)
            pooled_output = F.relu(pooled_output)
            logits = self.classifier(pooled_output)
            # calculate loss
            loss = 0
            if labels is not None:
                loss = self.loss_func(logits.view(-1, self.config['n_labels']), labels.view(-1,self.config['n_labels']))
            return loss, logits

        def training_step(self, batch, batch_index):
            loss, logits = self(**batch)
            self.log("train loss", loss, prog_bar = True, logger = True)
            return {"loss": loss, "predictions": logits, "labels": batch['labels']}

        def validation_step(self, batch, batch_index):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            loss, outputs = self(input_ids, attention_mask, labels=labels)
            self.log("validation loss ", loss, prog_bar = True, logger=True)
            return {"val_loss": loss, "predictions":outputs, "labels": labels}

        def test_step(self, batch, batch_index):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            loss, outputs = self(input_ids, attention_mask, labels=labels)
            self.log("test loss ", loss, prog_bar = True, logger=True)
            return loss

        def training_epoch_end(self, outputs):
            labels,predictions,roc =[],[],[]

            for output in outputs:
                for output_labels in output['labels'].detach().cpu():
                    labels.append(output_labels)
                for output_preds in output['predictions'].detach().cpu():
                    predictions.append(output_preds)
                    
            labels = torch.stack(labels)
            predictions = torch.sigmoid(torch.stack(predictions))

            for i, categories in enumerate(categories):
                roc_score = auroc(predictions[:, i], labels[:, i].int())
                roc.append(roc_score)
                self.logger.experiment.add_scalar(f"Train/{categories}_roc_auc", roc_score, self.current_epoch)
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

            for i, categories in enumerate(categories):
                roc_score = auroc(predictions[:, i], labels[:, i].int())
                roc.append(roc_score)
                self.logger.experiment.add_scalar(f"Validation/{attribute}_roc_auc", roc_score, self.current_epoch)
            self.log("ptl/val_avg_roc", np.mean(roc))

        def configure_optimizers(self):
            optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['w_decay'])
            total_steps = self.config['train_size'] / self.config['bs']
            warmup_steps = math.floor(total_steps * self.config['warmup'])
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
            return [optimizer], [scheduler]

    # data Module
    dataModule = DataModule(data_train, data_test, data_val, tokenizer)
    dataModule.setup()

    config = {
        'model_name':'distilbert-base-uncased', 
        'n_labels': n_categories,
        'bs': 16, 
        'lr': 5e-5,
        'pre_classifier':'relu',
        'warmup': 0.2,
        'train_size': len(dataModule.train_dataloader()),
        'w_decay': 0.001,
        'n_epochs': 15
    }

    model = SentenceClassifier(config)

    # train
    trainer = pl.Trainer(max_epochs=config['n_epochs'], num_sanity_val_steps=50)
    trainer.fit(model, dataModule)

    def classify_sentences(model, dm):
        predictions = trainer.predict(model, datamodule=dm)
        flattened_predictions = np.stack([torch.sigmoid(torch.Tensor(p)) for batch in predictions for p in batch])
        return flattened_predictions

    pred_labels = classify_sentences(model, dataModule)
    true_labels = np.array(data_test[categories])

    plt.figure()
    for i, categorie in enumerate(categories):
        fpr, tpr, _ = metrics.roc_curve(
            true_labels[:,i].astype(int), pred_labels[:, i])
        auc = metrics.roc_auc_score(
            true_labels[:,i].astype(int), pred_labels[:, i])
        plt.plot(fpr, tpr, label='%s %g' % (categorie, auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.title('AUC ROC')

if __name__ == '__main__':
    freeze_support()
    main()

# # def tokenize_function(example):
# #     return tokenizer(example['Sentence'], truncation=True, padding=True)

# # data = load_dataset("pandas", data_files="data\data_huggingface.pkl")
# # print("data", data['train'][0])
# # tokenized_data = data['train'].map(tokenize_function, batched=True)
# # tokenized_data = tokenized_data.train_test_split(test_size=0.2, shuffle=True)
# # print("tokenized_data", tokenized_data['train'][0])

# # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
# # tf_train_dataset = tokenized_data['train'].to_tf_dataset(
# #     columns=['attention_mask', 'input_ids'],
# #     label_cols=categories,
# #     shuffle=True,
# #     collate_fn=data_collator,
# #     batch_size=TRAIN_BATCH_SIZE
# # )

# # tf_test_dataset = tokenized_data['test'].to_tf_dataset(
# #     columns=['attention_mask', 'input_ids'],
# #     label_cols=categories,
# #     shuffle=False,
# #     collate_fn=data_collator,
# #     batch_size=TEST_BATCH_SIZE
# # )

# # print("tf_train_dataset", tf_train_dataset[0])
# # print("tf_test_dataset", tf_test_dataset[0])

# # # num_train_steps = len(tf_train_dataset) * EPOCHS
# # # lr_scheduler = PolynomialDecay(
# # #     initial_learning_rate=LEARNING_RATE, end_learning_rate=0.0, decay_steps=num_train_steps
# # # )
# # # loss = BinaryCrossentropy(from_logits=True)
# # # opt = Adam(learning_rate=lr_scheduler)
# # # model.compile(optimizer=opt, loss=loss, metrics=['accuracy', 'AUC'])
# # # model.fit(tf_train_dataset, validation_data=tf_test_dataset ,epochs=EPOCHS)
# # # y_pred = (model.predict(tf_test_dataset) > 0.5).astype(int)