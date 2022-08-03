# Create the predict function
from flask import Flask, jsonify, request, make_response
import numpy as np
import torch


# def predict(m, dm):
#     predictions = trainer.predict(m, dm)
#     flattened_predictions = np.stack([torch.sigmoid(torch.Tensor(p)) for batch in predictions for p in batch])
#     return flattened_predictions

# Code the flask APP

app = Flask(__name__)


@app.route('/', methods = ['POST'])
def result():
    message = request.form.get('message')
    print(message)
    # results = predict(message)
    #Prediction is used here
    #then you resturn a response with the values

if __name__ == '__main__':
    # starting app
    app.run(debug=True,host='0.0.0.0')