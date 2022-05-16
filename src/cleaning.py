import pandas as pd
import re
import string
import dvc.api as dvc
import nltk
import ssl
from nltk.corpus import stopwords

# // Functions

# Retirer punctuation, chiffres et mettre en minuscule
def cleanText(text):
    text = text.lower()
    text = text.strip() #removing whitespaces
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    return text

# Vérifier s'il y a des données manquantes
def checkMissingValues(df):
    nbEmptyCells = df.isna().sum().sum() 
    if (nbEmptyCells != 0):
        df.dropna(axis='rows')
    return(df)

# Could find an alternative for rows that have same sentence but different categories -> merge
nettoyage = lambda x: cleanText(x)

def getStopWords(): 
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    # Stop words
    nltk.download('stopwords')
    stop = stopwords.words('english')
    # Add stop word to array if necessary
    stopWords = ['id', 'im', 'ive']

    for word in stopWords:
        stop.append(word)

    return stop

cleaning_stopWords = lambda x: ' '.join([word for word in x.split() if word not in (getStopWords())])

# // Main 

with dvc.open(
        'data/data.xlsx',
        mode='rb'
        ) as fd:
        data = pd.read_excel(fd, sheet_name='Sheet1')     
data = data.astype('string') 

# Change column name
data.rename(columns={'ER/SF':'ER/SR'}, inplace=True)

# Make binary dataset
data['CP'] = data['CP'].apply(lambda x: 1 if x != "0" else 0)
data['Constructivism'] = data['Constructivism'].apply(lambda x: 1 if x != "0" else 0)
data['Coping'] = data['Coping'].apply(lambda x: 1 if x != "0" else 0)
data['ED'] = data['ED'].apply(lambda x: 1 if x != "0" else 0)
data['ER/SR'] = data['ER/SR'].apply(lambda x: 1 if x != "0" else 0)
data['Empowerment'] = data['Empowerment'].apply(lambda x: 1 if x != "0" else 0)
data['Exposure/Habit'] = data['Exposure/Habit'].apply(lambda x: 1 if x != "0" else 0)
data['PERMA'] = data['PERMA'].apply(lambda x: 1 if x != "0" else 0)

# Nettoyage
data['Sentence'] = data['Sentence'].apply(nettoyage)
data['Source'] = data['Source'].apply(nettoyage)

checkMissingValues(data)

# Remove repeated sentences
data = data.drop_duplicates(subset="Sentence",keep="first").reset_index() 

#Remove stopWords
data['Sentence'] = data['Sentence'].apply(cleaning_stopWords)

# Output the cleaned dataset in xlsx format
data.to_excel('./output/output_cleaning.xlsx')