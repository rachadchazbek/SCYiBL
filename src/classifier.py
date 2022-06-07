import json
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve, multilabel_confusion_matrix, classification_report
import yaml
import dvc.api as dvc


## Functions ##
## TO DO 
##ADD MULTI-LABEL CHOICE

def modelChoice(num):
    # can add class_weight='balanced' for LinearSVC and LogisticRegression
    if num == 1:
        model = LogisticRegression()
    elif num == 2:
        model = MultinomialNB()
    elif num == 3:
        model = LinearSVC()

    return model 

def vectChoice(num, ngram):
    if num == 1:
        vect = CountVectorizer(ngram_range=(1, ngram))
    elif num == 2:
         vect = TfidfVectorizer(ngram_range=(1, ngram))

    return vect 

## MAIN ## 
with open("./params.yaml") as conf_file:
    params = yaml.safe_load(conf_file)
print(params)

with dvc.open(
        './data/output_cleaning.xlsx',
        mode='rb'
        ) as fd:
        data = pd.read_excel(fd, sheet_name='Sheet1')

# data = pd.read_excel(params['data']['dataset'])  

# Text preprocessing has already been done (see cleaning.py)
# Predicting sentences classification one category at a time
classes = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_classes = len(classes)

# Step 1: train-test split
category = params['data']['category'] # NOT NEEDED 
# instantiate a vectorizer -> Convert a collection of text documents to a matrix of token counts
vect = vectChoice(params['train']['vector'], params['train']['ngram'])
# DTM = document term matrix = frequency of term in sentences
X = vect.fit_transform(data.Sentence) # the column text contains textual data to extract features from
y = data[classes] # this is the columns we are learning to predict.

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=params['train']['random_state'], test_size=params['train']['test_size'])

# Step 2: Train the classifier and predict for test data
classifier = modelChoice(params['train']['model'])
multiLabelModel = BinaryRelevance(classifier)
multiLabelModel.fit(X_train, y_train) # fit the model with training data
y_pred = multiLabelModel.predict(X_test) # Make predictions on test data
y_predict = pd.DataFrame.sparse.from_spmatrix(y_pred, columns=classes) # Make y_pred format similar to y_test


# Step 3: Evaluate the classifier using various measures
## Confusion Matrices ## 
# Create a plot with all the confusion matrix and put it in dvc pipeline ?
confusionMatrixCP = multilabel_confusion_matrix(y_test, y_pred)[0]
print(confusionMatrixCP)
score = classification_report(y_test, y_predict, output_dict=False, target_names=classes, zero_division=0)
print(score) # NOT SURE IF WE KEEP CLASSIFICATION REPORT

## Precision Recall Curve ##
precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i], recall[i], threshold = precision_recall_curve(y_test.iloc[:,i], y_predict.iloc[:,i])
    plt.plot(recall[i], precision[i], lw=2, label=f'{classes[i]}')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="lower left")
plt.title("precision vs. recall curve")
plt.savefig('output/pr_curve.png', format="png")

# ## Metrics ##
# dictionnary = { 
#     "auc": roc_auc_score(y_test, y_predict),
#     "f1_score": f1_score(y_test, y_predict),
#     "accuracy": accuracy_score(y_test, y_predict)
# }

# metrics_json = json.dumps(dictionnary, indent = 4)
with open("output/metrics.json", "w") as outfile:
    outfile.write('{}')


## Metrics ##

## ROC Curve ##
fpr = dict()
tpr = dict()
for i in range(n_classes):
    fpr[i], tpr[i], thresholds = roc_curve(y_test.iloc[:,i], y_predict.iloc[:,i])
    plt.plot(fpr[i], tpr[i], lw=2, label=f'{classes[i]}')
plt.title('ROC curve') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('output/roc_curve.png', format="png")


# csv classification report
# split curves into 2 figures
# auc and accuracy
# confusion matrix w/ 8 figures
