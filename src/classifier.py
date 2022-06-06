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

## Functions ##
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

data = pd.read_excel(params['data']['dataset'])  

# Text preprocessing has already been done (see cleaning.py)
# Predicting sentences classification one category at a time
classes = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_classes = len(classes)

# Step 1: train-test split
category = params['data']['category']
# instantiate a vectorizer -> Convert a collection of text documents to a matrix of token counts
vect = vectChoice(params['train']['vector'], params['train']['ngram'])
# DTM = document term matrix = frequency of term in sentences
X = vect.fit_transform(data.Sentence).toarray() # the column text contains textual data to extract features from
y = data[classes] # this is the columns we are learning to predict.

# Step 2: Split data
# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=params['train']['random_state'], test_size=params['train']['test_size'])

# Step 3: Train the classifier and predict for test data
classifier = modelChoice(params['train']['model'])
multiLabelModel = BinaryRelevance(classifier)
multiLabelModel.fit(X_train, y_train) # fit the model with training data
y_pred = multiLabelModel.predict(X_test) # Make predictions on test data

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

# Create a plot with all the confusion matrix and put it in dvc pipeline ?
confusionMatrixCP = multilabel_confusion_matrix(y_test, y_pred)[0]
print(confusionMatrixCP)
score = classification_report(y_test, y_pred, output_dict=False, target_names=classes, zero_division=0)
print(score)

'''
# Step 4: Evaluate the classifier using various measures
cmdf = pd.DataFrame({'actual': [y_test], 'predicted': [y_pred]})
cmdf.to_csv('./output/confusion_matrix.csv', index=False)
'''
print(y_test[:,5])
## Precision Recall Curve ##
## Get access to col ##
precision = []
recall = []
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test.loc[:, i], y_pred.loc[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()

'''
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
'''