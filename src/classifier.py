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
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
import yaml

## Functions ##
def modelChoice(num):
    # can add class_weight='balanced' for LinearSVC and LogistricRegression
    if num == 1:
        model = LogisticRegression(class_weight='balanced')
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

data = pd.read_excel(params['data']['dataset'])  

# Text preprocessing has already been done (see cleaning.py)
# Predicting sentences classification one category at a time
classes = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']


# Step 1: train-test split
category = params['data']['category']
X = data.Sentence # the column text contains textual data to extract features from
y = data[category] # this is the column we are learning to predict. 

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=params['train']['random_state'], test_size=params['train']['test_size'])

# Step 2: Preprocess and Vectorize train and test data
# instantiate a vectorizer -> Convert a collection of text documents to a matrix of token counts
vect = vectChoice(params['train']['vector'], params['train']['ngram'])
# DTM = document term matrix = frequency of term in sentences
X_train_dtm = vect.fit_transform(X_train) # use it to extract features from training data
# transform testing data (using training data's features)
X_test_dtm = vect.transform(X_test)
# Step 3: Train the classifier and predict for test data
model = modelChoice(params['train']['model'])
model.fit(X_train_dtm, y_train) # fit the model with training data
y_pred = model.predict(X_test_dtm) # Make predictions on test data


# Step 4: Evaluate the classifier using various measures
# have a csv file with y_test, y_pred
mapping = {i: cls_name for i, cls_name in enumerate(classes)}
cmdf = pd.DataFrame(
    {'actual': y_test, 'predicted': y_pred})
cmdf.to_csv('./output/confusion_matrix.csv', index=False)

## Precision Recall Curve ##
precision, recall, prc_thresholds = precision_recall_curve(y_test, y_pred)
nth_point = math.ceil(len(prc_thresholds) / 1000)
prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]
prc_file = os.path.join("output", "precision_recall.json")
with open(prc_file, "w") as fd:
    fd.write(json.dumps(
        {
            "prc": [
                {"precision": float(p), "recall": float(r), "threshold": int(t)}
                for p, r, t in prc_points
            ]
        },
        indent=4,
    ))

## Metrics ##
dictionnary = { 
    "auc": roc_auc_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "accuracy": accuracy_score(y_test, y_pred)
}

metrics_json = json.dumps(dictionnary, indent = 4)
with open("output/metrics.json", "w") as outfile:
    outfile.write(metrics_json)

## ROC Curve ##
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr)
plt.title('ROC curve') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('output/roc_curve.png', format="png")