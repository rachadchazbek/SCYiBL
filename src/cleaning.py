import pandas as pd
import re
import string
import dvc.api as dvc
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

# Could find an alternative for rows that have same sentence but different categories -> merge
nettoyage = lambda x: cleanText(x)

# Vérifier s'il y a des données manquantes
def checkMissingValues(df):
    nbEmptyCells = df.isna().sum().sum() 
    if (nbEmptyCells != 0):
        df.dropna(axis='rows')
    return(df)

def getStopWords(): 
    # Stop words
    stop = stopwords.words('english')
    # Add stop word to array if necessary
    # Stopwords added after 1st round of EDA (life, things, felt) because they were in 
    # top 20 words of every category
    stopWords = ['id', 'im', 'ive', 'life', 'things', 'felt']

    for word in stopWords:
        stop.append(word)
    return stop
    
words = getStopWords()
cleaning_stopWords = lambda x: ' '.join([word for word in x.split() if word not in (words)])

# Main 
with dvc.open(
        'data/data.xlsx',
        mode='rb'
        ) as fd:
        data = pd.read_excel(fd, sheet_name='Sheet1')     

# Change column name
data.rename(columns={'ER/SF':'ER/SR'}, inplace=True)


# Make binary dataset
data['CP'] = data['CP'].apply(lambda x: 1 if x != 0 else 0)
data['Constructivism'] = data['Constructivism'].apply(lambda x: 1 if x != 0 else 0)
data['Coping'] = data['Coping'].apply(lambda x: 1 if x != 0 else 0)
data['ED'] = data['ED'].apply(lambda x: 1 if x != 0 else 0)
data['ER/SR'] = data['ER/SR'].apply(lambda x: 1 if x != 0 else 0)
data['Empowerment'] = data['Empowerment'].apply(lambda x: 1 if x != 0 else 0)
data['Exposure/Habit'] = data['Exposure/Habit'].apply(lambda x: 1 if x != 0 else 0)
data['PERMA'] = data['PERMA'].apply(lambda x: 1 if x != 0 else 0)

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