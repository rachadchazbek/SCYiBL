import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

data = pd.read_excel('./output/output_cleaning.xlsx')    

# Exploratory Data Analysis - afficher mot les plus courants
wordCount_df = data['Sentence'].str.split(expand=True).stack().value_counts().reset_index()
wordCount_df.columns = ['Word', 'Frequency']
#print(wordCount_df) - to see wordcount for all the sentences

# Créer texte pour chaque catégorie
def textPerCategory(df, col):
    text = ""
    for i in range(len(df)): 
        # Ajouter chaque phrase au texte de la colonne
        if df.loc[i, col] == 1:
            text += df.loc[i,'Sentence']
            text += " "
    return text

# Appliquer pour chaque colonne
textCP = textPerCategory(data, 'CP')
textConstructivism = textPerCategory(data,'Constructivism')
textCoping = textPerCategory(data, 'Coping')
textED = textPerCategory(data, 'ED')
textER = textPerCategory(data, 'ER/SR')
textEmpowerment = textPerCategory(data, 'Empowerment')
textExposure = textPerCategory(data, 'Exposure/Habit')
textPERMA = textPerCategory(data, 'PERMA')

# faire un dictionaries avec key= catégorie and values = text
textDict = {'CP': textCP, 'Constructivism': textConstructivism, 'Coping': textCoping, 
        'ED': textED, 'ER': textER, 'Empowerment':textEmpowerment, 
        'Exposure/Habit': textExposure, 'PERMA':textPERMA}

# token and wordcount pour chaque key
for key, value in textDict.items():
    globals()[f'wordCount{key}'] = pd.value_counts(np.array(word_tokenize(value))).nlargest(20)

# display word frequency for each category
# word cloud

# histograms
wordCountED.plot(kind='bar')
plt.show()

# We can see that there is a lot of overlap in most used words per category 