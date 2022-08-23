print("IN main")
import torch
import pandas as pd
import numpy as np
import json
import pytorch_lightning as pl
from flask import Flask, request, make_response
from transformers import RobertaTokenizer
from dataModule import DataModule, DataModulePredict
from sklearn.model_selection import train_test_split
from sentenceClassifier import SentenceClassifier


app = Flask(__name__)
model_name = 'roberta-base'
categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_categories = len(categories)
tokenizer = RobertaTokenizer.from_pretrained(model_name)
data_path = "./data/data_huggingface.csv"


def split_data(data_path):
    data = pd.read_csv(data_path)
    data = data.drop(columns=['index'])
    x_train, x_testAndVal = train_test_split(data, test_size=0.2, shuffle=True)
    x_test, x_val = train_test_split(x_testAndVal, test_size=0.33,shuffle=True)
    return x_train, x_test, x_val


def createPositiveWeights(data_path):
    df = pd.read_csv(data_path)
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

    return neg_pos_ratio


data_train, data_test, data_val = split_data(data_path)
data_module = DataModule(data_pred=None, data_train=data_train, data_test=data_test, data_val=data_val,
                         tokenizer=tokenizer, batch_size=32)
data_module.setup()
config = {
    "lr": 1.5e-5,
    "batch_size": 10,
    "max_token_len": 128,
    "n_epochs": 0,
    "weight_decay": 0.001,
    "n_classes": n_categories,
    "warmup": 0.3,
    "dropout": False,
    "train_size": len(data_module.train_dataloader()),
    "pre_classifier": 'relu',
    "pool_method": 'mean',
    "class_weight": createPositiveWeights(data_path),
    'sample_weight': True,
    "optim":'sgd'
}

model = SentenceClassifier(config)

trainer = pl.Trainer(
    max_epochs=config['n_epochs'], 
    num_sanity_val_steps=10, 
    logger=False, 
    enable_checkpointing=False,
    enable_model_summary=None
)
trainer.fit(model, data_module)


def predict(m, dm):
    predictions = trainer.predict(m, datamodule=dm)
    flattened_predictions = np.stack([torch.sigmoid(torch.Tensor(p)) for batch in predictions for p in batch])
    return flattened_predictions


def string_to_dataframe(sentence):
    s_nd_array=np.array([["app", str(sentence), 0, 0, 0, 0, 0, 0, 0, 0]])
    dataframe=pd.DataFrame(s_nd_array, columns=['Source', 'Sentence', *categories])
    return dataframe


def dataframe_toDataModule(pd: pd.DataFrame):
    s_data_module = DataModulePredict(data_pred=pd, data_train=None, data_test=None, data_val=None,
                               tokenizer=tokenizer, batch_size=1)
    s_data_module.setup()
    return s_data_module


@app.route('/', methods = ['POST'])
def result():
    message = request.form['message']
    df = string_to_dataframe(message)
    dm = dataframe_toDataModule(df)
    output = predict(model, dm)
    print("Message :", message)
    print("Output :", output)
    output_json = json.dumps({r[0]: str(round(r[1], 2)) for r in zip(categories, output[0])}, indent=4)
    response = make_response(output_json, 200)
    return response


if __name__ == '__main__':
    # starting app
    app.run(debug=True, host='0.0.0.0', port=5010)
    print("Server is running")