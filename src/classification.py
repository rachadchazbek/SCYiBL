import pandas as pd
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.metrics import AUC
from keras.losses import BinaryCrossentropy
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

xls = pd.read_excel("./output/output_cleaning.xlsx") 
columns = ('CP', 'Constructivism', 'Coping', 'ED', 'ER/SF', 'Empowerment', 'Exposure/Habit', 'PERMA')
sentences = xls['Sentence']
rows = ['1', '0', '%']

# TODO move classes and classes_bin to evaluate file
classes = ('CP', 'Constructivism', 'Coping', 'ED', 'ER/SF', 'Empowerment', 'Exposure/Habit', 'PERMA')
classes_bin = ('CPb', 'Constructivismb', 'Copingb', 'EDb', 'ER/SFb', 'Empowermentb', 'Exposure/Habitb', 'PERMAb')

y = pd.concat([xls['CPb'], ['Constructivismb'], xls['Copingb'],  xls['EDb'],
               xls['ER/SFb'], xls['Empowermentb'], xls['Exposure/Habitb'], xls['PERMAb']],
              axis=1)

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

# Tokenize words
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

# Main settings
# TODO: Should be variables in each experiment USING DVC 
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


#TODO move to a new evaluate.py file
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
