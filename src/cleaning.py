import io
import pandas as pd
import spacy
import re
import string
import dvc.api as dvc
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# use remote repository instead of file path
with dvc.open(
        'data/data.xlsx',
        mode='rb'
        ) as fd:
        data = pd.read_excel(fd, sheet_name='Sheet1')     
data = data.astype(str) # dont know if it's useful


# retirer punctuation, chiffres et mettre en minuscule
def cleanText(text):
    text = text.lower()
    text = text.strip() #removing whitespaces
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    return text

nettoyage = lambda x: cleanText(x)
data['Sentence'] = data['Sentence'].apply(nettoyage)
data['Source'] = data['Source'].apply(nettoyage)

# Vérifier s'il y a des données manquantes
def checkMissingValues(df):
    nbEmptyCells = df.isna().sum().sum() # somme totale de cellule vide
    print(f"Nb of Empty Cells: {nbEmptyCells}")
    if (nbEmptyCells != 0):
        df.dropna(axis='rows') # for now, if there is at least one NA -> remove rows
    
    return(df)

checkMissingValues(data)

# Retirer les lignes ou les phrases sont répétées
data = data.drop_duplicates(subset="Sentence",keep="first").reset_index() # remove repeated sentences
'''
Could find an alternative for rows that have same sentence but different categories -> merge
'''

# retirer stop words
stop = stopwords.words('english')
# add stop word to stop if necessary
# example:
stop.append('id')
data['Sentence'] = data['Sentence'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print(data['Sentence'])


'''
# Théories mal encodées (out of range)
# pour chaque colonne, faire un set avec tout les chiffres, vérifier le range avec between
data_try = data_try['PERMA'].to_list()
data_set = set(",".join(str(e) for e in data_try))
data_set.remove(".")
data_set.remove(",")
data_set.remove(" ")
data_set = set(map(int, data_set))
print(sorted(data_set))
# checker si les sous-catégories sont dans le range indiqué
for i in range(len(sorted(data_set))):
    if 
'''


# Exploratory Data Analysis - afficher mot les plus courants pour chaque catégorie


# ESSAI
#for index, sentence in data['Sentence'].items():
#    print(f"Index: {index} is: {sentence}")
#print(data['Sentence'].sort_values(key=lambda x: x.str.lower()))
# montrer les phrases qui sont présentes plus d'une fois dans le dataset
#print(data[data.duplicated(subset=['Sentence'])]['Sentence'])

# similarité des phrases + doublons 
# if similarity = 1.0 -> doublons -> merge values des différentes catégories
'''
nlp = spacy.load("en_core_web_lg")
doc1 = nlp("miserable.")
doc2 = nlp("miserable.")
print(doc1.similarity(doc2))
'''