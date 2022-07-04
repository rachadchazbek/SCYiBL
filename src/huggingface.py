from datasets import DatasetDict, load_dataset
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
import tensorflow as tf
from tensorflow.data import Dataset 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from keras.losses import BinaryCrossentropy
from keras.metrics import AUC, accuracy
import pandas as pd 

class sentencesDataset(Dataset):
    
    def __init__(self, data_path, tokenizer, categories, max_len):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.categories = categories
        self.max_len = max_len
        self._prepare_data()

    def _prepare_data(self):
        data = load_dataset("pandas", data_files=self.data_path)
    
    def _len__(self):
        return(len(self.data))
    
    def _getitem__(self, index):
        item = self.data.iloc[index]
        sentence = str(item.Sentence)
        categories = tf.Tensor(item[self.categories])
        tokens = self.tokenizer.encode_plus(sentence, 
                                            add_special_tokens=True, 
                                            return_tensors='tf',
                                            truncation=True,
                                            max_length=self.max_len, 
                                            padding='max_length',
                                            return_attention_mask=True)
        return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': categories}

# distilled BERT 
categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_categories = len(categories)

# Parameters
MAX_LEN = 30
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 5e-5

modelName = "distilbert-base-uncased"
model = TFAutoModelForSequenceClassification.from_pretrained(modelName)
tokenizer = AutoTokenizer.from_pretrained(modelName)
dataset = sentencesDataset("data\data_huggingface.pkl", tokenizer, categories, max_len=30)

print(dataset._getitem__(0))

# def tokenize_function(example):
#     return tokenizer(example['Sentence'], truncation=True, padding=True)

# #data = load_dataset("pandas", data_files="data\data_huggingface.pkl")
# tokenized_data = data['train'].map(tokenize_function, batched=True)
# tokenized_data = tokenized_data.train_test_split(test_size=0.2, shuffle=True)

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
# tf_train_dataset = tokenized_data['train'].to_tf_dataset(
#     columns=['attention_mask', 'input_ids'],
#     label_cols=classes,
#     shuffle=True,
#     collate_fn=data_collator,
#     batch_size=TRAIN_BATCH_SIZE
# )

# tf_test_dataset = tokenized_data['test'].to_tf_dataset(
#     columns=['attention_mask', 'input_ids'],
#     label_cols=classes,
#     shuffle=False,
#     collate_fn=data_collator,
#     batch_size=TEST_BATCH_SIZE
# )

# num_train_steps = len(tf_train_dataset) * EPOCHS
# lr_scheduler = PolynomialDecay(
#     initial_learning_rate=LEARNING_RATE, end_learning_rate=0.0, decay_steps=num_train_steps
# )
# loss = BinaryCrossentropy(from_logits=True)
# opt = Adam(learning_rate=lr_scheduler)
# model.compile(optimizer=opt, loss=loss, metrics=['accuracy', 'AUC'])
# model.fit(tf_train_dataset, validation_data=tf_test_dataset ,epochs=EPOCHS)
# y_pred = (model.predict(tf_test_dataset) > 0.5).astype(int)