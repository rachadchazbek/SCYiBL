import pandas as pd
import spacy
import re
import string

# use remote repository instead of file path
xls = pd.ExcelFile(r"C:/Users/theob/Documents/data.xlsx") #use r before absolute file path
data = xls.parse(0) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis
sentences = data['Sentence']
classes = ('CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA')
classes_bin = ('CPb', 'Constructivismb', 'Copingb', 'EDb', 'ER/SRb', 'Empowermentb', 'Exposure/Habitb', 'PERMAb')

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
data_try = data.drop_duplicates(subset="Sentence",keep="first") # remove repeated sentences
'''
Could find an alternative for rows that have same sentence but different categories -> merge
'''

# retirer stop words
# Théories mal encodées (out of range)
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
