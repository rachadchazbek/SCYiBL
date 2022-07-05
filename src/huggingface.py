import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import auroc
import math
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, AdamW, get_cosine_schedule_with_warmup, AutoTokenizer
from sklearn.model_selection import train_test_split
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

modelName = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(modelName)
tokenizer = AutoTokenizer.from_pretrained(modelName)

class sentencesDataset(Dataset):

    def __init__(self, data_path, tokenizer, categories, max_token_len: int = 30):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.categories = categories
        self.max_token_len = max_token_len
        self._prepare_data()

    def _prepare_data(self):
        data = pd.read_pickle(self.data_path)
        self.data = data
    
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
        sentence = str(item.Sentence)
        categories = torch.FloatTensor(item[self.categories])
        tokens = self.tokenizer.encode_plus(sentence, 
                                            add_special_tokens=True, 
                                            return_tensors='pt',
                                            truncation=True,
                                            max_length=self.max_token_len, 
                                            padding='max_length',
                                            return_attention_mask=True)
        return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': categories}

ds = sentencesDataset("data\data_huggingface.pkl", tokenizer, categories)
train, test = train_test_split(ds, test_size=0.2)


class DataModule(pl.LightningDataModule):

    def __init__(self, data_path, categories, batch_size: int = 8, max_token_len: int = 30, model_name ="distilbert-base-uncased"):
        super().__init__()
        self.data_path = data_path
        self.categories = categories
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup(self, stage = None):
        if stage in (None, "fit"):
            self.train_dataset = train # unsure
            self.test_dataset = test # unsure
        if stage == 'predict':
            self.test_dataset = test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=1, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1, shuffle=False)

dataModule = DataModule("data\data_huggingface.pkl", categories)
dataModule.setup()
dl = dataModule.train_dataloader()
print(len(dl))

class SentenceClassifier(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.pretrained_model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], return_dict=True)
        self.hidden - nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        self.classification - nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_label'])
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
        self.dropout = nn.Dropout() # prevents/reduces overfitting


def forward(self, input_ids, attention_mask, labels=None):
    # distilled BERT model
    output = self.pretrained_model(input_ids = inputs_ids, attention_mask = attention_mask)
    pooled_output = torch.mean(output.last_hidden_state, 1)
    # nn classification layers
    pooled_output = self.hidden(pooled_output)
    pooled_output = self.dropout(pooled_output)
    pooled_output = F.relu(pooled_output)
    logits = self.classifier(pooled_output)
    # claculate loss
    loss = 0
    if labels is not None:
        loss = self.loss_func(logits.views(-1, self.config['n_labels']), labels.views(-1,self.config['n_labels']))
    return loss, logits

# Parameters
# MAX_LEN = 30
# TRAIN_BATCH_SIZE = 32
# TEST_BATCH_SIZE = 32
# EPOCHS = 15
# LEARNING_RATE = 5e-5

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