import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import itertools
import csv
import seaborn as sns

data = pd.read_excel('./output/output_cleaning.xlsx')    

# Exploratory Data Analysis - afficher mot les plus courants
wordCount_df = data['Sentence'].str.split(expand=True).stack().value_counts().reset_index()
wordCount_df.columns = ['Word', 'Frequency']
# Garder les 30 mots les plus utilisés
wordCount_df = wordCount_df.nlargest(30, 'Frequency')
wordCount_df.plot(kind='bar', x='Word', title="Total word count")
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
    # Normalize word count (divide by nb of words per category and in %)
    globals()[f'normalized_wordCount{key}'] = globals()[f'wordCount{key}'].div(len(value)/100)

# histograms
# Figure avec les 8 plots - valeur absolue
plt.suptitle("Word count per category", fontsize=16)
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

# Figure avec les 8 plots - valeur normalisée
plt.suptitle("Word count per category (Normalized)", fontsize=16)
plt.subplot(4, 2, 1)
plt.subplots_adjust(hspace=1)
plt.title("CP", y = 0.6)
normalized_wordCountCP.plot(kind='bar')
plt.subplot(4, 2, 2)
plt.title("Constructivism", y = 0.6)
normalized_wordCountConstructivism.plot(kind='bar')
plt.subplot(4, 2, 3)
plt.title("Coping", y = 0.6)
normalized_wordCountCoping.plot(kind='bar')
plt.subplot(4, 2, 4)
plt.title("ED", y = 0.6)
normalized_wordCountED.plot(kind='bar')
plt.subplot(4, 2, 5)
plt.title("ER/SR", y = 0.6)
normalized_wordCountER.plot(kind='bar')
plt.subplot(4, 2, 6)
plt.title("Empowerment", y = 0.6)
normalized_wordCountEmpowerment.plot(kind='bar')
plt.subplot(4, 2, 7)
plt.title("Exposure", y = 0.6)
normalized_wordCountExposure.plot(kind='bar')
plt.subplot(4, 2, 8)
plt.title("PERMA", y = 0.6)
normalized_wordCountPERMA.plot(kind='bar')
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
#print(commonAll)

# 1st Round EDA:
# life, things and felt are 3 words present in top 20 of every categories -> added to stopwords

# Trouver les mots qui sont uniques à chaque catégorie
def complementaryList(list1, array1):
    arrayCopy = array1.copy()
    if list1 in arrayCopy:
        arrayCopy.remove(list1)
    
    compList = list(itertools.chain.from_iterable(arrayCopy))
    return compList

# Cette fonction prend 2 listes comme argument et retourne 
# Les éléments qui sont seulement dans la liste 1
def uniqueElements(list1, list2):
    return list(set(list1).difference(list2))

arrayWordList = [wordListCP, wordListConstructivism, wordListCoping, wordListED,
                wordListER, wordListEmpowerment, wordListExposure, wordListPERMA]

# Créer liste complémentaire pour chaque catégorie
compListCP = complementaryList(wordListCP,arrayWordList)
compListConstructivism = complementaryList(wordListConstructivism,arrayWordList)
compListCoping = complementaryList(wordListCoping,arrayWordList)
compListED = complementaryList(wordListED,arrayWordList)
compListER = complementaryList(wordListER,arrayWordList)
compListEmpowerment = complementaryList(wordListEmpowerment,arrayWordList)
compListExposure = complementaryList(wordListExposure,arrayWordList)
compListPERMA = complementaryList(wordListPERMA,arrayWordList)

# Comparer la liste à sa liste complémentaire pour trouver les 
# mots uniques à chaque catégorie
uniqueCP = uniqueElements(wordListCP, compListCP)
uniqueConstructivism = uniqueElements(wordListConstructivism, compListConstructivism)
uniqueCoping = uniqueElements(wordListCoping, compListCoping)
uniqueED = uniqueElements(wordListED, compListED)
uniqueER = uniqueElements(wordListER, compListER)
uniqueEmpowerment = uniqueElements(wordListEmpowerment, compListEmpowerment)
uniqueExposure = uniqueElements(wordListExposure, compListExposure)
uniquePERMA = uniqueElements(wordListPERMA, compListPERMA)

uniqueDict = {'CP': uniqueCP, 'Constructivism': uniqueConstructivism, 'Coping': uniqueCoping, 
        'ED': uniqueED, 'ER': uniqueER, 'Empowerment':uniqueEmpowerment, 
        'Exposure': uniqueExposure, 'PERMA':uniquePERMA}

# Save words unique to each category in a file
#with open('uniqueWords.csv', 'w') as output:
#    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
#    writer.writerow(['Show words in top 20 that are specific to the category'])
#    for key, value in uniqueDict.items():
#        writer.writerow([key, value])

# Longueur des phrases
sentenceLength = data['Sentence'].str.len()
nbins = [0,50,100,150,200]
plt.hist(sentenceLength, nbins, histtype ='bar', edgecolor='black', align='right')
plt.title('Sentences length')
plt.xlabel('Nb of char')
plt.ylabel('Frequency')
plt.show()

# Longueur des phrases par catégorie (see if it's an even distribution)
# créer un tableau contenant les phrases 
# attribués à la catégorie (colonne)
def sentenceArrayPerCategory(df, col):
    sentences = []
    for i in range(len(df)): 
        # Ajouter chaque phrase au texte de la colonne
        if df.loc[i, col] == 1:
            sentences.append(df.loc[i,'Sentence'])
    return sentences

# Prend en entrée un tableau de string et
# retourne un tableau contenant le nbr de char
# de chaque string dans le même ordre
def nbChar(stringList):
    res = []
    for e in stringList:
        res.append(len(e))
    
    return res


# Appliquer pour chaque colonne
sentencesLengthCP = nbChar(sentenceArrayPerCategory(data, 'CP'))
sentencesLengthConstructivism = nbChar(sentenceArrayPerCategory(data,'Constructivism'))
sentencesLengthCoping = nbChar(sentenceArrayPerCategory(data, 'Coping'))
sentencesLengthED = nbChar(sentenceArrayPerCategory(data, 'ED'))
sentencesLengthER = nbChar(sentenceArrayPerCategory(data, 'ER/SR'))
sentencesLengthEmpowerment = nbChar(sentenceArrayPerCategory(data, 'Empowerment'))
sentencesLengthExposure = nbChar(sentenceArrayPerCategory(data, 'Exposure/Habit'))
sentencesLengthPERMA = nbChar(sentenceArrayPerCategory(data, 'PERMA'))

fig, ax = plt.subplots(4,2, sharex='col')
fig.subplots_adjust(hspace=0.4)
fig.suptitle('Sentences length per category', fontsize=20)
fig.supxlabel("Number of characters", fontsize=18)
fig.supylabel("Frequency", fontsize=18)
ax[0][0].set_title("CP")
ax[0][0].hist(sentencesLengthCP, nbins, histtype ='bar', edgecolor='black', align='right')
ax[1][0].set_title("Coping")
ax[1][0].hist(sentencesLengthCoping, nbins, histtype ='bar', edgecolor='black', align='right')
ax[2][0].set_title("ER")
ax[2][0].hist(sentencesLengthER, nbins, histtype ='bar', edgecolor='black', align='right')
ax[3][0].set_title("Exposure")
ax[3][0].hist(sentencesLengthExposure, nbins, histtype ='bar', edgecolor='black', align='right')
ax[0][1].set_title("Constructivism")
ax[0][1].hist(sentencesLengthConstructivism, nbins, histtype ='bar', edgecolor='black', align='right')
ax[1][1].set_title("ED")
ax[1][1].hist(sentencesLengthED, nbins, histtype ='bar', edgecolor='black', align='right')
ax[2][1].set_title("Empowerment")
ax[2][1].hist(sentencesLengthEmpowerment, nbins, histtype ='bar', edgecolor='black', align='right')
ax[3][1].set_title("PERMA")
ax[3][1].hist(sentencesLengthPERMA, nbins, histtype ='bar', edgecolor='black', align='right')
plt.show()