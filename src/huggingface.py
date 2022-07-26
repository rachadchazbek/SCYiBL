from multiprocessing import freeze_support
from re import A
from regex import R
import torch
import math
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torchmetrics.functional.classification import auroc
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
from transformers import RobertaTokenizer, RobertaModel, DistilBertModel, AdamW, DistilBertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import evaluate 
from sklearn import metrics
import pandas as pd 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_categories = len(categories)
model_name = 'roberta-base' #'distilbert-base-uncased-finetuned-sst-2-english'

class sentencesDataset(Dataset):

    def __init__(self, data: pd.DataFrame, tokenizer: RobertaTokenizer, max_token_len: int = 30):
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

        def __init__(self, data_train, data_test, data_val, tokenizer, batch_size = 32, max_token_len = 128):
            super().__init__()
            self.data_train = data_train
            self.data_test = data_test
            self.data_val = data_val
            self.batch_size = batch_size
            self.max_token_len = max_token_len
            self.tokenizer = tokenizer

        def setup(self, stage=None):
            self.train_dataset = sentencesDataset(self.data_train, self.tokenizer)
            self.test_dataset = sentencesDataset(self.data_test, self.tokenizer)
            self.val_dataset = sentencesDataset(self.data_val, self.tokenizer)
            print('train_dataset')
            print(self.train_dataset)
            print('test dataset')
            print(self.test_dataset)
            print('val dataset')
            print(self.val_dataset) 

        def train_dataloader(self):
            print('using train dataloader')
            print(self.train_dataset.__len__())
            return DataLoader(self.train_dataset,batch_size=self.batch_size,num_workers=2,shuffle=False)
        def val_dataloader(self):
            print('using val dataloader')
            print(self.val_dataset.__len__())
            return DataLoader(self.val_dataset,batch_size=self.batch_size,num_workers=1)
        def predict_dataloader(self):
            print('using test dataloader')
            print(self.test_dataset.__len__())
            return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=1, shuffle=False)
        def test_dataloader(self):
            print('using test dataloader')
            print(self.test_dataset.__len__())
            return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=2, shuffle=False)

class SentenceClassifier(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # distilBert Layer
        self.bert = RobertaModel.from_pretrained(model_name, return_dict = True)
        # pre-classifier layer
        if self.config['pre_classifier'] is not None:
            self.pre_classifier = nn.Linear(self.bert.config.hidden_size, 768)
            nn.init.xavier_uniform_(self.pre_classifier.weight)
        else:
            self.classifier = nn.Linear(self.bert.config.hidden_size, self.config['n_classes'])
            nn.init.xavier_uniform_(self.classifier.weight)
        # drop outs
        if self.config['dropout']:
            self.dropout = torch.nn.Dropout(0.3)
        # classifier layer
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
        self.log("train loss", loss, prog_bar = True, logger = True)
        return {"loss": loss, "predictions": outputs, "labels": batch["labels"]}

    def validation_step(self, batch, batch_index):
        loss, outputs = self(**batch)
        self.log("validation loss", loss, prog_bar = True, logger = True)
        return {"val loss": loss, "predictions": outputs, "labels": batch["labels"]}

    def test_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_index):
        _, logits = self(**batch)
        return logits

    def training_epoch_end(self, outputs):
        labels,predictions, roc =[],[],[]
        # voir les données sur lesquels le model s'entraine

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
        x_train, x_testAndVal = train_test_split(data, test_size=0.2, shuffle=True)
        x_test, x_val = train_test_split(x_testAndVal, test_size=0.33,shuffle=True)
        return x_train, x_test, x_val
    
    def createPositiveWeights(data_path):
        df = pd.read_pickle(data_path)
        categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
        labels = df[categories]
        pos = []
        neg = []
        N = len(df)
        for k, label in enumerate(labels):
            pos.append(sum(df[label] == 1))
            neg.append(N - pos[k])
        neg_pos_ratio = []

        for i in range(len(pos)):
            ratio = neg[i]/pos[i]
            neg_pos_ratio.append(ratio)
        print(neg_pos_ratio)

        return neg_pos_ratio

    data_train, data_test, data_val = split_data("data/data_huggingface.pkl")
    print('data train')
    print(data_train)
    print('data test')
    print(data_test)
    print('data val')
    print(data_val)

    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    print('data module')
    data_module = DataModule(data_train, data_test, data_val, tokenizer, batch_size=32)
    print('data setup')
    data_module.setup()
    
    config = {
        "lr": 1.5e-5,
        "batch_size": 32,
        "max_token_len": 128,
        "n_epochs": 25,
        "weight_decay":0.01,
        "n_classes":n_categories,
        "warmup":0.3,
        "dropout":False,
        "train_size": len(data_module.train_dataloader()),
        "pre_classifier":'relu',
        "pool_method":'mean',
        "class_weight": createPositiveWeights("data/data_huggingface.pkl"),
        'sample_weight': True,
        "optim":'sgd'
    }

    # Modeling
    print('model loading')
    model = SentenceClassifier(config)
    print('trainer')
    trainer = pl.Trainer(max_epochs=config['n_epochs'], num_sanity_val_steps=10, log_every_n_steps=10, logger=True)
    checkpoint_path = 'output/preTrainedModel.ckpt'
    print('trainer fit')
    trainer.fit(model, data_module) #ckpt_path=checkpoint_path
    trainer.save_checkpoint(checkpoint_path)
    #print("***trainer test***")
    # trainer.test(model, data_module.test_dataloader())
    print("***trainer validate***")
    trainer.validate(model, data_module.val_dataloader())

    # true labels 
    true_labels = np.array(data_test[categories])

    # pred labels 
    def predict(m, dm):
        predictions = trainer.predict(m, dm)
        flattened_predictions = np.stack([torch.sigmoid(torch.Tensor(p)) for batch in predictions for p in batch])
        return flattened_predictions
    
    print('time to predict')
    pred_labels = predict(model, data_module)
        
    print('labels outside loop')
    print(true_labels[:,0])
    print(pred_labels[:,0])

    y_pred = pred_labels.round().astype(int)

    for i, categorie in enumerate(categories):
        print('%s true labels in loop' % (categorie))
        print(true_labels[:,i])
        print('%s pred labels in loop' % (categorie))
        print(pred_labels[:,i])
        print("%s y_pred label" % (categorie))
        print(y_pred[:,i])
        fpr, tpr, _ = metrics.roc_curve(true_labels[:,i], pred_labels[:,i])
        auc = metrics.roc_auc_score(true_labels[:,i], pred_labels[:, i])
        plt.plot(fpr, tpr, lw=2, label='%s %g' % (categorie, auc))
        
        accuracy = metrics.accuracy_score(true_labels[:,i], pred_labels[:, i].round(), normalize=True)
        print('%s accuracy : ' % (categorie), accuracy)
    # show other metrics
    print(metrics.classification_report(true_labels, y_pred,target_names=categories))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="best")
    plt.title('ROC curve') 
    plt.show()

    for i, categorie in enumerate(categories):
        recall, precision, _ = metrics.precision_recall_curve(true_labels[:,i], pred_labels[:,i])
        plt.plot(recall, precision, lw=2, label='%s' % (categorie))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="best")
    plt.title('Precision-Recall Curve') 
    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()

# positive/negative ratio for each categories
# [0.8195232690124858, 9.210191082802547, 4.745519713261649, 3.0582278481012657, 0.48977695167286245, 4.088888888888889, 30.431372549019606, 3.1102564102564103]
# [1, 5, 3, 2, 1, 2, 10, 2]
# up constructivism from 5 to 9
# up exposure/habit from 10 to 15