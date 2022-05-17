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
# TO DO add axis title
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

# Transform wordCount Series to Dataframe (easier to merge)
# And creat list of top20 words for each category
for key, value in textDict.items():
    globals()[f'wordCount{key}'] = globals()[f'wordCount{key}'].rename_axis('Word').reset_index(name='Counts')
    globals()[f'wordList{key}'] = globals()[f'wordCount{key}']['Word'].tolist()

# Trouver les mots qui sont dans les top 20 de toutes les catégories 
# Cette fonction prend 2 listes comme argument et retourne 
# une liste contenant les éléments communs des 2 listes
def commonElements(list1, list2):
    return list(set(list1) & set(list2))

# there should be a faster way to compare 8 lists 
# 1st round of comparison
commonCP_Constructivism = commonElements(wordListCP, wordListConstructivism)
commonCoping_ED = commonElements(wordListCoping, wordListED)
commonER_Empowerment = commonElements(wordListER, wordListEmpowerment)
commonExposure_PERMA = commonElements(wordListExposure, wordListPERMA)

# 2nd round of comparison
commonCP_Cons_Coping_ED = commonElements(commonCP_Constructivism, commonCoping_ED)
commonER_Empow_Exposure_PERMA = commonElements(commonER_Empowerment,commonExposure_PERMA)

# 3rd round
commonAll = commonElements(commonCP_Cons_Coping_ED,commonER_Empow_Exposure_PERMA)
print(commonAll)

# life, things and felt are 3 words present in top 20 of every categories -> add to stopwords
# We can see that there is a lot of overlap in most used words per category 