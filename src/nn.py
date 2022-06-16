from ast import Num
import gensim
from gensim.models import Word2Vec
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, auc, f1_score, roc_auc_score
import tqdm
import yaml
import numpy as np
import dvc.api as dvc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from sklearn.model_selection import train_test_split, RepeatedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Flatten, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from keras.initializers import Constant
from keras.metrics import Recall, AUC, accuracy
import pickle

# Model architecture (What layers to use ? How many layers to use ? How many units for each layer ?)
## Feature eng (Vectorization, word count per category, sentiment analysis, correlation between category)
# Cross validation 
# Grid Search

## Constants
DATA_PATH = './data/output_cleaning.xlsx'

## FUNCTIONS ## 
def vectChoice(num, ngram):
    if num == 1:
        vect = CountVectorizer(ngram_range=(1, ngram))
    elif num == 2:
         vect = TfidfVectorizer(ngram_range=(1, ngram))

    return vect 

def createModel(num_filters, kernel_size, embedding_dim, maxlen, vocab_size, embedding_matrix):
    model = Sequential()

    # possibilities to also used pre-trained word embedding like GLoVE or word2Vec
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen, trainable=False, weights=[embedding_matrix]))
    model.add(Dropout(0.1))
    # Note: 
    # To connect a Dense layer directly to an Embedding layer
    # you must first flatten the 2D output matrix to a 1D vector using the Flatten layer
    # num_filters and kernel_size needs fine tuning 
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='relu'))
    # final layer of our neural network will be a dense layer 
    # with a sigmoid activation consisting of units equivalent to the number of classes.
    model.add(Dense(8, activation='sigmoid'))
    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=[AUC(multi_label=True, num_labels=8, from_logits=False), Recall()])
    return model

# Count unique words
def wordCounter(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count 

def loadEmbedding(file):
    # load the whole embedding into memory
    print('loading word embeddings...')
    embeddings_index = dict()
    f = open(file, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    return embeddings_index 

def createEmbeddingMatrix(vocab_size, embedding_dim, embedding_index):
    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    words_not_found = [] # List of words without embeddings
    for word, i in wordIndex.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print("Number of words not found: ", len(words_not_found))

    return embedding_matrix

## MAIN ## 
with open("./params.yaml") as conf_file:
    params = yaml.safe_load(conf_file)['NeuralNetwork']

maxlen = params['maxlen'] 
epochs = params['epochs']
embedding_dim = params['embedding_dim'] 
batch_size = params['batch_size']
kernel_size = params['kernel_size']
num_filters = params['num_filters']

with dvc.open(
        DATA_PATH,
        mode='rb'
        ) as fd:
        data = pd.read_excel(fd, sheet_name='Sheet1')

# Text preprocessing has already been done (see cleaning.py)
classes = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_classes = len(classes)

# Step 1: train-test split 
X = data.Sentence # the column text contains textual data to extract features from
y = data[classes] # this is the columns we are learning to predict.

#cv = RepeatedKFold(n_splits=5, random_state=params['train']['random_state'])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=params['train']['random_state'], test_size=params['train']['test_size'])

counter = wordCounter(X)
num_words = len(counter) # 3545 different words overall
vocab_size = num_words+1

# Main parameters 


tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(X_train)
# word_index creates a dict with every word assigned to a value
# Words are indexed by descending order of count
wordIndex = tokenizer.word_index

# Merge every word index to form a sentence of indexes
trainSentences = tokenizer.texts_to_sequences(X_train)
# Set maximum length to 10 (under -> completed with 0s, over -> truncated)
# Training set
X_train = pad_sequences(trainSentences, padding='post', truncating='post', maxlen=maxlen)
# Test set
testSentences = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(testSentences, padding='post', truncating='post', maxlen=maxlen)
 
# Word embedding w/ pre-trained model 
# Use next 4 lines when we create embbedingMatrix for the 1st time, from a doc downloaded 
#embeddingsIndex = loadEmbedding('crawl-300d-2M.vec')
#embeddingMatrix = createEmbeddingMatrix(vocab_size, embedding_dim, embeddingsIndex)
#with open('output/GloveEmbMatrix.pkl', 'wb') as file:
#    pickle.dump(embeddingMatrix, file)

# Use next 2 lines to load pre-existing embedded Matrices (present in output repository)
#with open('output/GloveEmbMatrix.pkl', 'rb') as embMatrixFile:
#    embeddingMatrix = pickle.load(embMatrixFile)


# ## MODEL ##
model = createModel(num_filters, kernel_size, embedding_dim, maxlen, vocab_size, embeddingMatrix)
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

with open('output/model.pkl', 'wb') as file:
    pickle.dump(model, file)
    
## EVALUATE ## 
y_pred = (model.predict(X_test) > 0.5).astype(int)
y_testArray = np.array(y_test)
f1_score = f1_score(y_pred, y_testArray, average='weighted')
#auc = roc_auc_score(y_pred, y_testArray, average='weighted'),
accuracy = accuracy_score(y_pred, y_testArray)
#print("auc: ", auc)
print("Accuracy: ", accuracy)
print("f1_score: ", f1_score)

# Graph of accuracy and loss over epochs