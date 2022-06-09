from ast import Num
import pandas as pd
import yaml
import dvc.api as dvc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.initializers import Constant


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
# instantiate a vectorizer -> Convert a collection of text documents to a matrix of token counts
vect = vectChoice(params['train']['vector'], params['train']['ngram'])
# DTM = document term matrix = frequency of term in sentences
X = data.Sentence # the column text contains textual data to extract features from
# vect.fit_transform()
y = data[classes] # this is the columns we are learning to predict.

# Test/Train split
# split X and y into training and testing sets
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
reverse_wordIndex = dict([(value, key) for (key, value) in wordIndex.items()])
def decode(text):
    return " ".join([reverse_wordIndex.get(i, "?") for i in text])


# Merge every word index to form a sentence of indexes
# Set maximum length to 10 (under -> completed with 0s, over -> truncated)
# Training set
X_train = pad_sequences(trainSentences, padding='post', truncating='post', maxlen=maxlen)

# For test set
testSentences = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(testSentences, padding='post', truncating='post', maxlen=maxlen)

## MODEL ##
model = Sequential()

model.add(Embedding(num_words, 32, input_length=maxlen))
model.add(LSTM(64, dropout=0.1))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

model.summary()
epochs = 10
history = model.fit(X_train, y_train["CP"], epochs=epochs, validation_data=(X_test, y_test["CP"]))