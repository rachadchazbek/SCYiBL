# Create the predict function
from multiprocessing import freeze_support
from xml.dom.minidom import TypeInfo
from flask import Flask, request, make_response
import torch
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer
import pytorch_lightning as pl
from dataModule import DataModule
from sklearn.model_selection import train_test_split
from sentenceClassifier import SentenceClassifier

app = Flask(__name__)
model_name = 'roberta-base'
categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_categories = len(categories)
checkpoint_path = './output/preTrainedModel.ckpt'
tokenizer = RobertaTokenizer.from_pretrained(model_name)


def split_data(data_path):
    data = pd.read_pickle(data_path)
    data = data.drop(columns=['index'])
    x_train, x_testAndVal = train_test_split(data, test_size=0.2, shuffle=True)
    x_test, x_val = train_test_split(x_testAndVal, test_size=0.33,shuffle=True)
    print(x_test, type(x_test))
    return x_train, x_test, x_val

def createPositiveWeights(data_path):
    df = pd.read_pickle(data_path)
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

data_train, data_test, data_val = split_data("data/data_huggingface.pkl")
data_module = DataModule(None , data_train, data_test, data_val, tokenizer, batch_size=32)
data_module.setup()
config = {
    "lr": 1.5e-5,
    "batch_size": 10,
    "max_token_len": 128,
    "n_epochs": 1,
    "weight_decay":0.001,
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

model = SentenceClassifier(config)

trainer = pl.Trainer(
    max_epochs=config['n_epochs'], 
    num_sanity_val_steps=10, 
    log_every_n_steps=10, 
    logger=True, 
    checkpoint_callback=False, 
)
trainer.fit(model, data_module)

def predict(m, dm):
    trainer.validate(model, dm.predict_dataloader())
    predictions = trainer.predict(m, datamodule=dm)
    flattened_predictions = np.stack([torch.sigmoid(torch.Tensor(p)) for batch in predictions for p in batch])
    return flattened_predictions

# Code the flask APP
def string_to_dataframe(sentence):
    s_nd_array=np.array([["app",sentence, 0, 0, 0, 0, 0, 0, 0, 0 ]])
    dataframe=pd.DataFrame(s_nd_array, columns=['Source', 'Sentence', *categories])
    print(s_nd_array, dataframe)
    return dataframe

def dataframe_toDataModule(pd: pd.DataFrame):
    s_data_module = DataModule(data_pred=pd ,data_train= None, data_test = None, data_val=None,tokenizer=tokenizer, batch_size=1)
    s_data_module.setup(predict)
    print(s_data_module.data_pred)
    return s_data_module

@app.route('/', methods = ['POST'])
def result():
    message = request.form['message']
    df = string_to_dataframe(message)
    dm =  dataframe_toDataModule(df)
    print(dm, type(dm))
    output = predict(model, dm)
    print(message, output)
    # results = predict(message)
    #Prediction is used here
    #then you resturn a response with the assigned categories
    response = make_response(output, 200)
    return response

if __name__ == '__main__':
    # starting app
    app.run(debug=True,host='0.0.0.0')
    print("Server is running")