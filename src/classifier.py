import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, auc, roc_curve
import yaml

## Functions ##
def modelChoice(num):
    # can add class_weight='balanced' for LinearSVC and LogistricRegression
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


# Step 1: train-test split
X = data.Sentence # the column text contains textual data to extract features from
y = data.CP # this is the column we are learning to predict. 

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

#cf_matrix = confusion_matrix(y_test, y_pred)
    # Accuracy = what proportion of observation our system labeled correctly (TP+TN/TP+FP+TN+FN)
accuracy = accuracy_score(y_test, y_pred)
    # Precision = what proportion of the items our system labeled positive is actually positive (TP/TP+FP)
precision = precision_score(y_test, y_pred)
    # Recall = percentage of items actually present in input that were correctly labeled (TP/TP+FN)
recall = recall_score(y_test, y_pred)
f1_score = 2*precision*recall / (precision + recall)

fpr, tpr, threshold = roc_curve(y_test, y_pred)
auc = auc(fpr, tpr)

print(auc)
    # stats_text = "\n\nAccuracy={:0.2f}\nPrecision={:0.2f}\nRecall={:0.2f}\nF1 Score={:0.2f}".format(
    #                    accuracy,precision,recall,f1_score)
    #group_names = ['TN','FP','FN','TP']
    #group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    #group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    #labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    #labels = np.asarray(labels).reshape(2,2)
    #sns.heatmap(cf_matrix, annot=labels, fmt='')
    #if params['train']['model'] == 1:
    #    if params['train']['vector'] == 1:
    #        plt.title(f"{col}, Logistic Regression, CountVectorizer, {params['train']['ngram']}-gram")
    #    elif params['train']['vector'] == 2:
    #        plt.title(f"{col}, Logistic Regression, TFIDF vectorizer, {params['train']['ngram']}-gram")
    #elif params['train']['model'] == 2:
    #    if params['train']['vector'] == 1:
    #        plt.title(f"{col}, Multinomial NB, CountVectorizer, {params['train']['ngram']}-gram")
    #    elif params['train']['vector'] == 2:
    #        plt.title(f"{col}, Multinomial NB, TFIDF vectorizer, {params['train']['ngram']}-gram")
    #elif params['train']['model'] == 3:
    #    if params['train']['vector'] == 1:
    #        plt.title(f"{col}, Linear SVC, CountVectorizer, {params['train']['ngram']}-gram")
    #    elif params['train']['vector'] == 2:
    #        plt.title(f"{col}, Linear SVC, TFIDF vectorizer, {params['train']['ngram']}-gram")
    #plt.xlabel('Predicted Label' + stats_text)
    #plt.ylabel('True Label') 
    #plt.subplots_adjust(bottom=0.3)
    #plt.savefig(params['evaluate']['file_path'], format='png')