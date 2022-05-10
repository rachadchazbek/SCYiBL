import pandas as pd
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.metrics import AUC
from keras.losses import BinaryCrossentropy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, coverage_error

def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def create_multi_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(8, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss=BinaryCrossentropy(from_logits=True),
                  metrics=[AUC(multi_label=True, num_labels=8, from_logits=True)])
    return model

xls = pd.ExcelFile("data.xlsx") #use r before absolute file path
data = xls.parse(0) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis
sentences = data['Sentence']
classes = ('CP', 'Constructivism', 'Coping', 'ED', 'ER/SF', 'Empowerment', 'Exposure/Habit', 'PERMA')
classes_bin = ('CPb', 'Constructivismb', 'Copingb', 'EDb', 'ER/SFb', 'Empowermentb', 'Exposure/Habitb', 'PERMAb')

data = data.astype("string")
data['CPb'] = data['CP'].apply(lambda x: 1 if x != "0" else 0)
data['Constructivismb'] = data['Constructivism'].apply(lambda x: 1 if x != "0" else 0)
data['Copingb'] = data['Coping'].apply(lambda x: 1 if x != "0" else 0)
data['EDb'] = data['ED'].apply(lambda x: 1 if x != "0" else 0)
data['ER/SFb'] = data['ER/SF'].apply(lambda x: 1 if x != "0" else 0)
data['Empowermentb'] = data['Empowerment'].apply(lambda x: 1 if x != "0" else 0)
data['Exposure/Habitb'] = data['Exposure/Habit'].apply(lambda x: 1 if x != "0" else 0)
data['PERMAb'] = data['PERMA'].apply(lambda x: 1 if x != "0" else 0)

#y = data['ER/SFb']
y = pd.concat([data['CPb'], data['Constructivismb'], data['Copingb'], data['EDb'],
               data['ER/SFb'], data['Empowermentb'], data['Exposure/Habitb'], data['PERMAb']],
              axis=1)

def display_frequencies(data):
    nb_sentences = data['Sentence'].count()
    counts = [[data[data['CPb'] != 0]['CPb'].sum(), data[data['Constructivismb'] != 0]['Constructivismb'].sum(),
               data[data['Copingb'] != 0]['Copingb'].sum(), data[data['EDb'] != 0]['EDb'].sum(),
               data[data['ER/SFb'] != 0]['ER/SFb'].sum(), data[data['Empowermentb'] != 0]['Empowermentb'].sum(),
               data[data['Exposure/Habitb'] != 0]['Exposure/Habitb'].sum(), data[data['PERMAb'] != 0]['PERMAb'].sum()],
              [nb_sentences - data[data['CPb'] != 0]['CPb'].sum(), nb_sentences - data[data['Constructivismb'] != 0]['Constructivismb'].sum(),
               nb_sentences - data[data['Copingb'] != 0]['Copingb'].sum(), nb_sentences - data[data['EDb'] != 0]['EDb'].sum(),
               nb_sentences - data[data['ER/SFb'] != 0]['ER/SFb'].sum(), nb_sentences - data[data['Empowermentb'] != 0]['Empowermentb'].sum(),
               nb_sentences - data[data['Exposure/Habitb'] != 0]['Exposure/Habitb'].sum(), nb_sentences - data[data['PERMAb'] != 0]['PERMAb'].sum()]]

    columns = ('CP', 'Constructivism', 'Coping', 'ED', 'ER/SF', 'Empowerment', 'Exposure/Habit', 'PERMA')
    rows = ['1', '0', '%']
    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    n_rows = len(counts)
    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4
    y_offset = np.zeros(len(columns))
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, counts[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + counts[row]
        cell_text.append(['%1.1f' % x for x in counts[row]])
    cell_text.append([round(100*x/nb_sentences, 2) for x in counts[0]])
    # Reverse colors and text labels to display the last value at the top.
    #colors = colors[::-1]
    #cell_text.reverse()

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    values = np.arange(0, nb_sentences+1, 100)
    value_increment = 1
    #plt.ylabel("Loss in ${0}'s".format(value_increment))
    plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.xticks([])
    plt.title('Data')
    plt.savefig('data.png', dpi=300)

display_frequencies(data)

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

# Tokenize words
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

# Main settings
epochs = 5
embedding_dim = 50
maxlen = 100
output_file = 'output.txt'

# Pad sequences with zeros
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Parameter grid for grid search
param_grid = dict(num_filters=[32, 64, 128],
                  kernel_size=[3, 5, 7])
model = KerasClassifier(num_filters=32, kernel_size=7,
                        vocab_size=vocab_size,
                        maxlen=maxlen,
                        embedding_dim=embedding_dim,
                        model=create_multi_model,
                        epochs=epochs, batch_size=10,
                        verbose=True)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                          cv=4, verbose=1, n_iter=5)
grid_result = grid.fit(X_train, y_train)

# Evaluate testing set
test_accuracy = grid.score(X_test, y_test)
y_pred = grid.predict(X_test)
f1_scores = f1_score(y_test, y_pred, average=None)
print('F1-score : ', )
for i in range(len(classes)):
    print(classes[i], accuracy_score(y_test[classes_bin[i]], y_pred[:, i]))
print(coverage_error(y_test, y_pred))

# Save and evaluate results
with open(output_file, 'a') as f:
    output_string = f"""
Coverage error = {round(coverage_error(y_test, y_pred), 2)}
Global accuracy = {round(test_accuracy, 2)}
Accuracy per label :
"""
    for i in range(len(classes)):
        output_string += f"   {classes[i]} = {round(accuracy_score(y_test[classes_bin[i]], y_pred[:, i]), 2)}\n"
    output_string += "F1-scores :\n"
    for i in range(len(classes)):
        output_string += f"   {classes[i]} = {round(f1_scores[i], 2)}\n"
    print(output_string)
    f.write(output_string)