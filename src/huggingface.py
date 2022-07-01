from datasets import DatasetDict, load_dataset
from transformers import pipeline
from transformers import BertConfig, TFBertModel
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import pandas as pd 

## FUNCTIONS ##
modelName = "distilbert-base-uncased-finetuned-sst-2-english"
model = TFAutoModelForSequenceClassification.from_pretrained(modelName)
tokenizer = AutoTokenizer.from_pretrained(modelName)

def tokenize_function(example):
     return tokenizer(example['Sentence'], truncation=True)

data = load_dataset("pandas", data_files="data\data_huggingface.pkl")
tokenized_data = data['train'].map(tokenize_function, batched=True)
tokenized_data = tokenized_data.train_test_split(test_size=0.2, shuffle=True)
print(tokenized_data)

#tokenized_datasets = data['train']['Sentence'].map(tokenize_function, batched=True)
#print(tokenized_datasets[3])

# distilled BERT 
# # config = BertConfig()
# inputs = tokenizer(data['Sentence'][3], padding=True, truncation=True, return_tensors="tf")
# batch = dict(inputs)

# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
# labels = tf.convert_to_tensor([1, 1])
# model.train_on_batch(batch, labels)
# outputs = model(**inputs)

# # classifier = pipeline("zero-shot-classification")
# # classes = ['Cognitive Processing', 'Coping', 'Emotion recognition', 'Empowerment', 'PERMA']
# # res = classifier("Nevermind, I don't want to do this", candidate_labels=classes,  model=model, tokenizer=tokenizer)
# # print(res)