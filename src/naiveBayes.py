import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import feature extraction methods from sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# import classifiers from sklearn
from sklearn.naive_bayes import MultinomialNB

# import different metrics to evaluate the classifiers
from sklearn.metrics import accuracy_score, precision_score, recall_score

# from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn import metrics

data = pd.read_excel('./output/output_cleaning.xlsx')  

# Text preprocessing has already been done (see cleaning.py)
# Let's start by predicting 1st category of the dataset only (CP)
data = data[['Sentence', 'CP']]

# Step 1: train-test split
X = data.Sentence # the column text contains textual data to extract features from
y = data.CP # this is the column we are learning to predict. 

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.25)

# Step 2: Preprocess and Vectorize train and test data
vect = CountVectorizer() # instantiate a vectorizer -> Convert a collection of text documents to a matrix of token counts
# DTM = document term matrix = frequency of term in sentences
X_train_dtm = vect.fit_transform(X_train) # use it to extract features from training data
# transform testing data (using training data's features)
X_test_dtm = vect.transform(X_test)

# Step 3: Train the classifier and predict for test data
nb = MultinomialNB() # instantiate a Multinomial Naive Bayes model
nb.fit(X_train_dtm, y_train) # train the model
y_pred = nb.predict(X_test_dtm) # make class predictions for X_test_dtm


# Step 4: Evaluate the classifier using various measures
# Confusion Matrix
# Recall = percentage of items actually present in input that were correctly labeled (TP/TP+FN)
print("Recall: ", recall_score(y_test, y_pred))
# Precision = what proportion of the items our system labeled as 1 are actually 1 (TP/TP+FP)
print("Precision: ", precision_score(y_test, y_pred))
# Accuracy = what proportion of observation our system labeled correctly (TP+TN/TP+FP+TN+FN)
print("Accuracy: ", accuracy_score(y_test, y_pred)) 