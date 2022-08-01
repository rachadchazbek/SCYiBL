from multiprocessing import freeze_support
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd 
import matplotlib.pyplot as plt
from dataModule import DataModule
from sentenceClassifier import SentenceClassifier

categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_categories = len(categories)
model_name = 'roberta-base' #'distilbert-base-uncased-finetuned-sst-2-english'

def main(): 
    categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
    n_categories = len(categories)

    # Split dataset: train 80, test 20 + shuffle 
    def split_data(data_path):
        data = pd.read_pickle(data_path)
        data = data.drop(columns=['index'])
        x_train, x_testAndVal = train_test_split(data, test_size=0.2, shuffle=True)
        x_test, x_val = train_test_split(x_testAndVal, test_size=0.33,shuffle=True)
        return x_train, x_test, x_val
    
    def createPositiveWeights(data_path):
        df = pd.read_pickle(data_path)
        categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
        labels = df[categories]
        pos = []
        neg = []
        N = len(df)
        for k, label in enumerate(labels):
            pos.append(sum(df[label] == 1))
            neg.append(N - pos[k])
        neg_pos_ratio = []

        for i in range(len(pos)):
            ratio = neg[i]/pos[i]
            neg_pos_ratio.append(ratio)
        print(neg_pos_ratio)

        return neg_pos_ratio

    data_train, data_test, data_val = split_data("data/data_huggingface.pkl")
    print('data train')
    print(data_train)
    print('data test')
    print(data_test)
    print('data val')
    print(data_val)

    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    print('data module')
    data_module = DataModule(data_train, data_test, data_val, tokenizer, batch_size=32)
    print('data setup')
    data_module.setup()
    
    config = {
        "lr": 1.5e-5,
        "batch_size": 10,
        "max_token_len": 128,
        "n_epochs": 10,
        "weight_decay":0.001,
        "n_classes":n_categories,
        "warmup":0.3,
        "dropout":False,
        "train_size": len(data_module.train_dataloader()),
        "pre_classifier":'relu',
        "pool_method":'mean',
        "class_weight": createPositiveWeights("data/data_huggingface.pkl"),
        'sample_weight': True,
        "optim":'sgd'
    }

    # Modeling
    print('model loading')
    model = SentenceClassifier(config)
    print('trainer')
    trainer = pl.Trainer(max_epochs=config['n_epochs'], num_sanity_val_steps=10, log_every_n_steps=10, logger=True)
    checkpoint_path = 'output/preTrainedModel.ckpt'
    print('trainer fit')
    trainer.fit(model, data_module) #ckpt_path=checkpoint_path
    trainer.save_checkpoint(checkpoint_path)
    print("***trainer validate***")
    trainer.validate(model, data_module.val_dataloader())

    # true labels 
    true_labels = np.array(data_test[categories])

    # pred labels 
    def predict(m, dm):
        predictions = trainer.predict(m, dm)
        flattened_predictions = np.stack([torch.sigmoid(torch.Tensor(p)) for batch in predictions for p in batch])
        return flattened_predictions
    
    print('time to predict')
    pred_labels = predict(model, data_module)
        
    print('labels outside loop')
    print(true_labels[:,0])
    print(pred_labels[:,0])

    y_pred = pred_labels.round().astype(int)

    for i, categorie in enumerate(categories):
        print('%s true labels in loop' % (categorie))
        print(true_labels[:,i])
        print('%s pred labels in loop' % (categorie))
        print(pred_labels[:,i])
        print("%s y_pred label" % (categorie))
        print(y_pred[:,i])
        fpr, tpr, _ = metrics.roc_curve(true_labels[:,i], pred_labels[:,i])
        auc = metrics.roc_auc_score(true_labels[:,i], pred_labels[:, i])
        plt.plot(fpr, tpr, lw=2, label='%s %g' % (categorie, auc))
        
        accuracy = metrics.accuracy_score(true_labels[:,i], pred_labels[:, i].round(), normalize=True)
        print('%s accuracy : ' % (categorie), accuracy)
    # show other metrics
    print(metrics.classification_report(true_labels, y_pred,target_names=categories))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="best")
    plt.title('ROC curve') 
    plt.show()

    for i, categorie in enumerate(categories):
        recall, precision, _ = metrics.precision_recall_curve(true_labels[:,i], pred_labels[:,i])
        plt.plot(recall, precision, lw=2, label='%s' % (categorie))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="best")
    plt.title('Precision-Recall Curve') 
    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()

# positive/negative ratio for each categories
# [0.8195232690124858, 9.210191082802547, 4.745519713261649, 3.0582278481012657, 0.48977695167286245, 4.088888888888889, 30.431372549019606, 3.1102564102564103]
# [1, 5, 3, 2, 1, 2, 10, 2]
# up constructivism from 5 to 7
# up exposure/habit from 10 to 15