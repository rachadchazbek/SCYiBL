from ast import Num
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc
import yaml
import dvc.api as dvc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from keras.initializers import Constant

# Make it multilabel
# Good metrics
# Ameliorer le reseau lui même
## Vectorization
## word count 

def vectChoice(num, ngram):
    if num == 1:
        vect = CountVectorizer(ngram_range=(1, ngram))
    elif num == 2:
         vect = TfidfVectorizer(ngram_range=(1, ngram))

    return vect 


# Attempt at creating a neural network
## MAIN ## 
with open("./params.yaml") as conf_file:
    params = yaml.safe_load(conf_file)

with dvc.open(
        './data/output_cleaning.xlsx',
        mode='rb'
        ) as fd:
        data = pd.read_excel(fd, sheet_name='Sheet1')

# Text preprocessing has already been done (see cleaning.py)
classes = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_classes = len(classes)

# Step 1: train-test split 
X = data.Sentence # the column text contains textual data to extract features from
y = data[classes] # this is the columns we are learning to predict.

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=params['train']['random_state'], test_size=params['train']['test_size'])

# Count unique words
def wordCounter(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count 

counter = wordCounter(X)
num_words = len(counter)
# 3135 different words in training set 
# 3544 different words overall

# Max nb of words in a sentence (50% is 8, 75% is 11, max is 26)
maxlen = 10

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(X_train)
# word_index creates a dict with every word assigned to a value
# Words are indexed by descending order of count
wordIndex = tokenizer.word_index


trainSentences = tokenizer.texts_to_sequences(X_train)
# Merge every word index to form a sentence of indexes
# Set maximum length to 10 (under -> completed with 0s, over -> truncated)
# Training set
X_train = pad_sequences(trainSentences, padding='post', truncating='post', maxlen=maxlen)

# For test set
testSentences = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(testSentences, padding='post', truncating='post', maxlen=maxlen)

## MODEL ##
model = Sequential()

# possibilities to also used pre-trained word embedding like GLoVE or word2Vec
# output_dim needs fine tuning
model.add(Embedding(input_dim=num_words+1, output_dim=10, input_length=maxlen))
model.add(Dropout(0.1))
# Note: 
# To connect a Dense layer directly to an Embedding layer
# you must first flatten the 2D output matrix to a 1D vector using the Flatten layer
# num_filters and kernel_size needs fine tuning 
model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
model.add(GlobalMaxPooling1D())
# final layer of our neural network will be a dense layer 
# with a sigmoid activation consisting of units equivalent to the number of classes.
model.add(Dense(8, activation='sigmoid'))
model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC()])

model.summary()
epochs = 20
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))