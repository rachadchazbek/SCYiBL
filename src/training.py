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
model_name = 'roberta-base'

def main(): 
    def split_data(data_path):
        data = pd.read_pickle(data_path)
        data = data.drop(columns=['index'])
        x_train, x_testAndVal = train_test_split(data, test_size=0.2, shuffle=True)
        x_test, x_val = train_test_split(x_testAndVal, test_size=0.33,shuffle=True)
        return x_train, x_test, x_val
    
    def createPositiveWeights(data_path):
        df = pd.read_pickle(data_path)
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
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    data_module = DataModule(data_train, data_test, data_val, tokenizer, batch_size=32)
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
    checkpoint_path = 'output/preTrainedModel.ckpt'
    model = SentenceClassifier(config)
    trainer = pl.Trainer(max_epochs=config['n_epochs'], num_sanity_val_steps=10, log_every_n_steps=10, logger=True)
    trainer.fit(model, data_module) #ckpt_path=checkpoint_path
    trainer.save_checkpoint(checkpoint_path)
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

    s_pred_label = DataModule(predict_dataset="my string", tokenizer=tokenizer)

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
