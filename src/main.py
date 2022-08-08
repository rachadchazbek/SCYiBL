# Create the predict function
import string
from flask import Flask, jsonify, request, make_response
import torch
import pandas as pd
import numpy as np
from pytorch_lightning import LightningModule

app = Flask(__name__)
categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_categories = len(categories)
checkpoint_path = './output/preTrainedModel.ckpt'
model = torch.load(checkpoint_path)

# def predict(m, dm):
#     predictions = trainer.predict(m, dm)
#     flattened_predictions = np.stack([torch.sigmoid(torch.Tensor(p)) for batch in predictions for p in batch])
#     return flattened_predictions

# Code the flask APP
def string_to_dataframe(sentence: string):
    s_nd_array=np.array([["app",sentence, 0, 0, 0, 0, 0, 0, 0, 0 ]])
    dataframe=pd.DataFrame(s_nd_array, columns=['Source', 'Sentence', *categories])
    return dataframe

@app.route('/', methods = ['POST'])
def result():
    message = request.form['message']
    print(message)
    print(model)
    # results = predict(message)
    #Prediction is used here
    #then you resturn a response with the assigned categories
    response = make_response(message, 200)
    return response

if __name__ == '__main__':
    # starting app
    app.run(debug=True,host='0.0.0.0')