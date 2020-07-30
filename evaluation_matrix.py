###########################
## Author - Shuting Xing ##
## The following function is for showing the result of the model after training and predictions

import json
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

json_file_path = "config.json"
with open(json_file_path, 'r') as j:
     params = json.loads(j.read())
params.keys()

optuna_args_dict = params['optuna']
hps_dict = params['hparams']
data_args_list = params['data']



def evaluation_matrix_cnn(model, history):
    """
    This function plots a confusion matrix, classification report and training and testing accuracy and loss
    """
    # Confusion Matrix
    from dataloader import dataloader_cnn

    x_train, x_test, x_val, y_train, y_test, y_val = dataloader_cnn(data_args_list)

    pred  = model.predict_classes(x_test, verbose=1)
    cm = confusion_matrix(y_test, pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title("Confusion Matrix")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()
    
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Accuracy with Epochs
    ax1.set_title('Model Accuracy')
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend(['train', 'validation'], loc='upper left')

    # Loss with Epoch
    ax2.set_title('Model Loss')
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.legend(['train', 'validation'], loc='upper left')

    fig.set_size_inches(20, 5)
    plt.show()
    
    # Classification Report
    
    print(classification_report(y_true=y_test , y_pred=pred))

    
    
    
def evaluation_matrix_xgb(model):
    from dataloader import dataloader_xgb

    """
    This function plots a confusion matrix, classification report
    """
    x_train, x_test, x_val, y_train, y_test, y_val = dataloader_xgb(data_args_list)
    # Confusion Matrx
    pred  = model.predict(x_test)
    cm = confusion_matrix(y_test, pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title("Confusion Matrix")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()
    
    # Classification Report
    
    print(classification_report(y_true=y_test , y_pred=pred))
