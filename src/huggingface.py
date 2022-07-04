from datasets import DatasetDict, load_dataset
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from keras.losses import BinaryCrossentropy
from keras.metrics import AUC, accuracy
import pandas as pd 

# distilled BERT 
classes = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_classes = len(classes)

# Parameters
MAX_LEN = 30
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 5e-5


modelName = "distilbert-base-uncased"
model = TFAutoModelForSequenceClassification.from_pretrained(modelName, problem_type="multi_label_classification", num_labels=8) #num_labels=n_classes
tokenizer = AutoTokenizer.from_pretrained(modelName)

def tokenize_function(example):
    return tokenizer(example['Sentence'], truncation=True, padding=True)

data = load_dataset("pandas", data_files="data\data_huggingface.pkl")
tokenized_data = data['train'].map(tokenize_function, batched=True)
tokenized_data = tokenized_data.train_test_split(test_size=0.2, shuffle=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
tf_train_dataset = tokenized_data['train'].to_tf_dataset(
    columns=['attention_mask', 'input_ids'],
    label_cols=classes,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=TRAIN_BATCH_SIZE
)

tf_test_dataset = tokenized_data['test'].to_tf_dataset(
    columns=['attention_mask', 'input_ids'],
    label_cols=classes,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=TEST_BATCH_SIZE
)

num_train_steps = len(tf_train_dataset) * EPOCHS
lr_scheduler = PolynomialDecay(
    initial_learning_rate=LEARNING_RATE, end_learning_rate=0.0, decay_steps=num_train_steps
)
loss = BinaryCrossentropy(from_logits=True)
opt = Adam(learning_rate=lr_scheduler)
model.compile(optimizer=opt, loss=loss, metrics=['accuracy', 'AUC'])
model.fit(tf_train_dataset, validation_data=tf_test_dataset ,epochs=EPOCHS)
y_pred = (model.predict(tf_test_dataset) > 0.5).astype(int)