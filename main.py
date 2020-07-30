###########################
## Author - Shuting Xing ##
###########################


## The following function is for Experiment Manager with Keras and conducting study for tuned model and showing results
## Sanity Checks are in the constructor __init__
## Functions are separated and called in the main class


## models.py - contains the xgboost model and CNN model, the mdoel can be changed from the config.json file
#### The idea is to compile the model in the class only
## evaluationmatrix.py contains the results showing for the models such as Confusion Matrix and Clasification Report (containing Precision, Recall, F1 score) for XGboost and Model losses, Accuracy score for train and validation for each epoch as a plot, Confusion Matrix and Clasification Report (containing Precision, Recall, F1 score)
## dataloader.py has capaciity to load the data and create three sets.. since CNN and Xgboost, both needs the data into different sets.. 
## showresults.py - It shows the best parameters, number of trials and best trial for the optuna study performed for each model


# Base imports 
import sys
import datetime
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Machine Learning imports
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Deep Learning imports
import keras
print('Keras', keras.__version__)
import keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Flatten, Dense, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adagrad, RMSprop, Adam, Adadelta, Adamax, Nadam
from tensorflow.keras.backend import clear_session
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Nadam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint




# import Optuna and OptKeras after Keras
import optuna 
print('Optuna', optuna.__version__)
from optuna.integration import TFKerasPruningCallback
from optkeras.optkeras import OptKeras
import optkeras
print('OptKeras', optkeras.__version__)
# (Optional) Disable messages from Optuna below WARN level.
optuna.logging.set_verbosity(optuna.logging.WARN)



# Import the created functions
from models import XGB_optuna, CNN_optuna
from evaluation_matrix import evaluation_matrix_xgb, evaluation_matrix_cnn
from show_results import show_results
from dataloader import dataloader_xgb, dataloader_cnn


import os
try:
    os.mkdir('data')
    os.mkdir('data/train')
    os.mkdir('data/test')
    os.mkdir('data/val')
except FileExistsError:
    print("Directory already exists")
    
    
json_file_path = "config.json"
with open(json_file_path, 'r') as j:
     params = json.loads(j.read())
params.keys()

# Optuna Parameters from config.json
study_name = params['optuna']['name']
optim_direction = params['optuna']['direction']
db = params['optuna']['db']
accuracy = params['optuna']['metric']

# Parameters for CNN model
hp_filters = params['hparams']["hp_filters"]['values']
hp_kernel_size = params['hparams']["hp_kernel_size"]['values']
hp_strides = params['hparams']["hp_strides"]['values']
hp_activation = params['hparams']["hp_activation"]['values']
hp_lrmin = params['hparams']["hp_lrmin"]['values']
hp_lrmax = params['hparams']["hp_lrmax"]['values']


# Parameters for XGboost Model

n_estimators = params['hparams']['n_estimators']['values']
max_depth = params['hparams']['max_depth']['values']
min_child_weight = params['hparams']['min_child_weight']['values']
learning_rate = params['hparams']['learning_rate']['values']
scale_pos_weight = params['hparams']['scale_pos_weight']['values']
subsample = params['hparams']['subsample']['values']
colsample_bytree = params['hparams']['colsample_bytree']['values']

    
# Data Parameters
data_path = params['data']['data_path']
log_path = params['data']['log_path']
model_path = params['data']['model_path']

optuna_args_dict = params['optuna']
hps_dict = params['hparams']
data_args_list = params['data']

# SQLITE Connectiivity
import sqlite3
con = sqlite3.connect(optuna_args_dict['name'] + '_Optuna.db')


callbacks_list = []
log_dir="logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks_list.append(tensorboard_callback)
w_fn = 'models\\mnist-1-{}.h5'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
callbacks_list.append(early_stopping)
# Change the cooldown to 1, if behances unexpectedly
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=2.5e-5, cooldown=0)
callbacks_list.append(learning_rate_reduction)
model_checkpoint = ModelCheckpoint(w_fn, monitor='val_accuracy', save_best_only=True)
callbacks_list.append(model_checkpoint)


class ExperimentManager(object):
    def __init__(self, model, optuna_args_dict, hps_dict, data_args_list, callbacks_list):
        
        self.optuna_args_dict = optuna_args_dict
        self.hps_dict = hps_dict
        self.callbacks_list = callbacks_list
        self.data_args_list = data_args_list
        self.model = model
        
    def __call__(self, trial):
        
        if optuna_args_dict['model'] == 'XGBOOST':
            
            
            #### Load the data from Dataloader for XGBOOST 
            x_train, x_test, x_val, y_train, y_test, y_val = dataloader_xgb(data_args_list)
            #### Calling XGBOOST model 
            #### Compile model, train and test in the main class
            model = XGB_optuna(hps_dict, trial)
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            #### Calculating Accuracy
            acc_score = accuracy_score(y_test, preds) 
            #### showing Evaluation
            evaluation_matrix_xgb(model)

        elif optuna_args_dict['model'] == "CNN":

            #### Load the data from Dataloader for CNN 
            x_train, x_test, x_val, y_train, y_test, y_val = dataloader_cnn(data_args_list)
            #### Calling CNN model 
            #### Compile model, train and test in the main class
            model = CNN_optuna(hps_dict, trial)
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_test, y_test),
                shuffle=True,
                batch_size=hps_dict['batch_size']['values'],
                epochs=hps_dict['epochs']['values'],
                verbose=2,
                callbacks=callbacks_list
            )

            # Evaluate the model accuracy on the test set.
            score = model.evaluate(x_val, y_val, verbose=2)
            #### Calculating Accuracy
            acc_score = score[1] 
            #### showing Evaluation
            evaluation_matrix_cnn(model, history)
        
        return acc_score
    
    
if __name__ == '__main__':  
    # Sanity Checks - Putting the Sanity check in the constructor, not in the call...
    if optuna_args_dict['db'] != "SQLite":
        raise TypeError("Invalid Database!, please use dbname SQLite")
    if optuna_args_dict['direction'] != "maximize":
        raise TypeError("Invalid Direction! please use direction as maximize")
    if optuna_args_dict['metric'] != "accuracy":
        raise TypeError("Invalid Metric! please use metric as accuracy")
    if optuna_args_dict['name'] != "NRC":  # Change to NRC, I already has one NRC named DB, also update in Config
        raise TypeError("Invalid Name! please use Study Name as NRC")
    
    model = optuna_args_dict['model']
    objective = ExperimentManager(model, optuna_args_dict, hps_dict, callbacks_list, data_args_list)
    study = optuna.create_study(study_name = study_name , direction=optuna_args_dict['direction'], storage='sqlite:///' +         optuna_args_dict['name'] + '_Optuna.db')
    study.optimize(objective, timeout=1*60) # Timeout in seconds e.g. timeout=20*60*60 
    show_results(study)
    # Load study
    #optuna.load_study(study_name=study_name, storage='sqlite:///' + optuna_args_dict['name'] + '_Optuna.db')
    #print(study.best_trial)
    