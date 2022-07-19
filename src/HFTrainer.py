from multiprocessing import freeze_support
from datasets import load_dataset
from regex import R
import torch
from torchmetrics.functional.classification import auroc
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl
from transformers import DataCollatorWithPadding, DistilBertModel, AdamW, DistilBertTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_categories = len(categories)
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'

class sentencesDataset(Dataset):

    def __init__(self, data: pd.DataFrame, tokenizer: DistilBertTokenizer, max_token_len: int = 30):
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

        def __init__(self, data_train, data_test, data_val, tokenizer, batch_size = 8, max_token_len = 128):
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
            return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=2, shuffle=False)
        def test_dataloader(self):
            print('using test dataloader')
            print(self.test_dataset.__len__())
            return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=2, shuffle=False)

def main(): 
    categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
    n_categories = len(categories)

    # Split dataset: train 80, test 20 + shuffle 
    def split_data(data_path):
        data = load_dataset("pandas", data_files=data_path)
        print("DATA: ", data)
        # data = data.drop(columns=['index'])
        x_train, x_testAndVal = train_test_split(data, test_size=0.2, shuffle=True)
        x_test, x_val = train_test_split(x_testAndVal, test_size=0.66,shuffle=True)
        return x_train, x_test, x_val

    data_train, data_test, data_val = split_data("data/data_huggingface.pkl")

    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name, return_dict=True)

    data_module = DataModule(data_train, data_test, data_val, tokenizer, batch_size=32)
    data_module.setup()

    training_args = TrainingArguments("test-trainer")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=data_module.train_dataloader(),
        eval_dataset=data_module.test_dataloader(),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    true_labels = np.array(data_test[categories])
    predictions = trainer.predict(data_module.val_dataloader())

    # # Converting to binary
    # to_binary = {True: 1, False: 0}
    # pred_labels = np.vectorize(to_binary.get)(pred_labels)
    # #true_labels = np.vectorize(to_binary.get)(true_labels)
    fpr = []
    tpr = []
    auc = []
    # print("pred labels", pred_labels[:,2])
    # print("true_labels", true_labels[:,2])
    for i in range(n_categories):
        fpr[i], tpr[i], _ = metrics.roc_curve(true_labels[:,i], predictions[:,i])
        auc[i] = metrics.roc_auc_score(true_labels[:,i], predictions[:, i])
        # roc_auc_score = evaluate.load("roc_auc")
        # auc[i] = roc_auc_score.compute(references=true_labels[:,i], prediction_scores= pred_labels[:, i])

        # precision = evaluate.load("precision")
        # accuracy[i] = precision.compute(references=true_labels[:,i], prediction_scores= pred_labels[:, i])

        # recall = evaluate.load("recall")
        # tpr[i] = recall.compute(references=true_labels[:,i], prediction_scores= pred_labels[:, i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{categories[i]}, {auc[i]}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="best")
    plt.title('ROC curve') 
    plt.show()
    # ADD SOME METRICS 
    print(metrics.classification_report(true_labels, predictions, target_names=categories))

if __name__ == '__main__':
    freeze_support()
    main()