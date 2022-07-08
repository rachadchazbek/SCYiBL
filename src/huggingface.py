from multiprocessing import freeze_support
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import auroc
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
from transformers import AutoModel, BertModel, AdamW, get_cosine_schedule_with_warmup, AutoTokenizer
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

categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_categories = len(categories)

class sentencesDataset(Dataset):

    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, max_token_len: int = 30):
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

        return {'sentence': sentence,
                'input_ids': encoding.input_ids.flatten(), 
                'attention_mask': encoding.attention_mask.flatten(), 
                'labels': torch.FloatTensor(labels)}

class DataModule(pl.LightningDataModule):

        def __init__(self, data_train, data_test, tokenizer, batch_size = 8, max_token_len = 30):
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
        def test_dataloader(self):
            return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=2, shuffle=False)

class SentenceClassifier(pl.LightningModule):

    def __init__(self, n_classes: int = n_categories, model_name = 'distilbert-base-uncased-finetuned-sst-2-english', n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name, return_dict=True)
        self.hidden = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
        self.dropout = nn.Dropout(0.2) # prevents/reduces overfitting


    def forward(self, input_ids, attention_mask, labels=None):
        # distilled BERT model
        output = self.model(input_ids = input_ids, attention_mask = attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        pooled_output = torch.mean(output.last_hidden_state, 1)
        # # nn classification layers
        pooled_output = self.hidden(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = F.relu(pooled_output)
        logits = self.classifier(pooled_output)
        # calculate loss
        loss = 0
        if labels is not None:
            loss = self.loss_func(output, labels)
        return loss, output

    def training_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train loss", loss, prog_bar = True, logger = True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val loss", loss, prog_bar = True, logger = True)
        return loss

    def test_step(self, batch, batch_index):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            loss, outputs = self(input_ids, attention_mask, labels)
            self.log("test loss", loss, prog_bar = True, logger = True)
            return loss

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
        #self.log("ptl/train_avg_roc", np.mean(roc))

#     def validation_epoch_end(self, outputs):
#         labels,predictions,roc =[],[],[]
        
#         for output in outputs:
#             for output_labels in output['labels'].detach().cpu():
#                 labels.append(output_labels)
#             for output_preds in output['predictions'].detach().cpu():
#                 predictions.append(output_preds)

#         labels = torch.stack(labels)
#         predictions = torch.sigmoid(torch.stack(predictions))

#         for i, categorie in enumerate(categories):
#             roc_score = auroc(predictions[:, i], labels[:, i].int())
#             roc.append(roc_score)
#             self.logger.experiment.add_scalar(f"Validation/{categorie}_roc_auc", roc_score, self.current_epoch)
#         self.log("ptl/val_avg_roc", np.mean(roc))

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1.5e-5)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.n_warmup_steps, num_training_steps=self.n_training_steps)
        return dict(optimizer=optimizer, lr_schedule=scheduler, interval='step')

def main(): 
    # Split dataset: train 70, val 10, test 20 + shuffle 
    def split_data(data_path):
        data = pd.read_pickle(data_path)
        data = data.drop(columns=['index'])
        x_train, x_test = train_test_split(data, test_size=0.2, shuffle=True)
        #x_val, x_test = train_test_split(x_valAndTest, test_size=0.66, shuffle=True)
        return x_train, x_test

    data_train, data_test = split_data("data\data_huggingface.pkl")
    categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
    n_categories = len(categories)

    model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, return_dict=True)

    EPOCHS = 10
    BATCH_SIZE = 8

    data_module = DataModule(data_train, data_test, tokenizer, batch_size=BATCH_SIZE)
    data_module.setup()

    # Modeling
    steps_per_epoch = len(data_train)
    total_training_steps = steps_per_epoch * EPOCHS
    warmup_steps = total_training_steps // 5 

    model = SentenceClassifier(n_classes=n_categories, n_warmup_steps=warmup_steps, n_training_steps=total_training_steps)
    trainer = pl.Trainer(max_epochs=EPOCHS, num_sanity_val_steps=50)
    trainer.fit(model, data_module)

# Predict 
# trained_model = SentenceClassifier.load_from_checkpoint("", n_classes=8)
#trained_model.freeze()


# def main():
#     def classify_sentences(model, dm):
#         predictions = trainer.predict(model, datamodule=dm)
#         flattened_predictions = np.stack([torch.sigmoid(torch.Tensor(p)) for batch in predictions for p in batch])
#         return flattened_predictions

#     pred_labels = classify_sentences(model, dataModule)
#     true_labels = np.array(data_test[categories])

#     plt.figure()
#     for i, categorie in enumerate(categories):
#         fpr, tpr, _ = metrics.roc_curve(
#             true_labels[:,i].astype(int), pred_labels[:, i])
#         auc = metrics.roc_auc_score(
#             true_labels[:,i].astype(int), pred_labels[:, i])
#         plt.plot(fpr, tpr, label='%s %g' % (categorie, auc))
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend(loc='best')
#     plt.title('AUC ROC')

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