import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

data = pd.read_excel('./output/output_cleaning.xlsx')    

# Exploratory Data Analysis - afficher mot les plus courants
wordCount_df = data['Sentence'].str.split(expand=True).stack().value_counts().reset_index()
wordCount_df.columns = ['Word', 'Frequency']
# Garder les 30 mots les plus utilisés
wordCount_df = wordCount_df.nlargest(30, 'Frequency')
wordCount_df.plot(kind='bar', x='Word')
plt.show()

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
        'Exposure': textExposure, 'PERMA':textPERMA}

# token and wordcount pour chaque key
for key, value in textDict.items():
    # return top 20 most used words for the category
    globals()[f'wordCount{key}'] = pd.value_counts(np.array(word_tokenize(value))).nlargest(20)

# display word frequency for each category
# word cloud

# histograms
# Figure avec les 8 plots
plt.subplot(4, 2, 1)
plt.subplots_adjust(hspace=1)
plt.title("CP", y = 0.6)
wordCountCP.plot(kind='bar')
plt.subplot(4, 2, 2)
plt.title("Constructivism", y = 0.6)
wordCountConstructivism.plot(kind='bar')
plt.subplot(4, 2, 3)
plt.title("Coping", y = 0.6)
wordCountCoping.plot(kind='bar')
plt.subplot(4, 2, 4)
plt.title("ED", y = 0.6)
wordCountED.plot(kind='bar')
plt.subplot(4, 2, 5)
plt.title("ER/SR", y = 0.6)
wordCountER.plot(kind='bar')
plt.subplot(4, 2, 6)
plt.title("Empowerment", y = 0.6)
wordCountEmpowerment.plot(kind='bar')
plt.subplot(4, 2, 7)
plt.title("Exposure", y = 0.6)
wordCountExposure.plot(kind='bar')
plt.subplot(4, 2, 8)
plt.title("PERMA", y = 0.6)
wordCountPERMA.plot(kind='bar')

plt.show()

# mettre en avant les mots qui sont dans plusieurs catégories 
# We can see that there is a lot of overlap in most used words per category 