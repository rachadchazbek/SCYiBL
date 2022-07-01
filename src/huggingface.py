from datasets import DatasetDict, load_dataset
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
from keras.optimizers import Adam, PolynomialDecay
from keras.losses import BinaryCrossentropy
from keras.metrics import AUC, accuracy
import pandas as pd 

# distilled BERT 
classes = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_classes = len(classes)

modelName = "distilbert-base-uncased-finetuned-sst-2-english"
model = TFAutoModelForSequenceClassification.from_pretrained(modelName, num_labels=n_classes)
tokenizer = AutoTokenizer.from_pretrained(modelName)

def tokenize_function(example):
    return tokenizer(example['Sentence'], truncation=True, padding=True)

data = load_dataset("pandas", data_files="data\data_huggingface.pkl")
tokenized_data = data['train'].map(tokenize_function, batched=True)
tokenized_data = tokenized_data.train_test_split(test_size=0.2, shuffle=True)

tf_train_dataset = tokenized_data['train'].to_tf_dataset(
    columns=['attention_mask', 'input_ids'],
    label_cols=classes,
    shuffle=True,
    batch_size=8
)

tf_test_dataset = tokenized_data['test'].to_tf_dataset(
    columns=['attention_mask', 'input_ids'],
    label_cols=classes,
    shuffle=True,
    batch_size=8
)

epochs = 15
num_train_steps = len(tf_train_dataset) * epochs
lr_scheduler = PolynomialDecay(
    initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps
)
loss = BinaryCrossentropy(from_logits=True)
opt = Adam(learning_rate=lr_scheduler)
model.compile(optimizer=opt, loss=loss, metrics=['accuracy', 'AUC'])
model.fit(tf_train_dataset, validation_data=tf_test_dataset ,epochs=epochs)
y_pred = (model.predict(X_test) > 0.5).astype(int)