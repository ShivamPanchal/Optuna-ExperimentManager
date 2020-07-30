###########################
## Author - Shuting Xing ##
## The following function contains models that we use in our experiment manager study



import numpy as np
import pandas as pd


from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Flatten, Dense, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adagrad, RMSprop, Adam, Adadelta, Adamax, Nadam
import keras.backend as K

import keras
print('Keras', keras.__version__)

# import Optuna and OptKeras after Keras
import optuna 
print('Optuna', optuna.__version__)
from optuna.integration import TFKerasPruningCallback

from optkeras.optkeras import OptKeras
import optkeras
print('OptKeras', optkeras.__version__)

# (Optional) Disable messages from Optuna below WARN level.
optuna.logging.set_verbosity(optuna.logging.WARN)


from tensorflow.keras.backend import clear_session
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Nadam
import sys
import datetime
import json
import itertools
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import xgboost as xgb

json_file_path = "config.json"
with open(json_file_path, 'r') as j:
     params = json.loads(j.read())
params.keys()

optuna_args_dict = params['optuna']
hps_dict = params['hparams']
data_args_list = params['data']


def XGB_optuna(hps_dict, trial):
    
    # param_list

    n_estimators = trial.suggest_categorical('n_estimators', hps_dict['n_estimators']['values'])
    max_depth = trial.suggest_categorical('max_depth', hps_dict['max_depth']['values'])
    min_child_weight = trial.suggest_categorical('min_child_weight', hps_dict['min_child_weight']['values'])
    learning_rate = trial.suggest_categorical('learning_rate', hps_dict['learning_rate']['values'])
    scale_pos_weight = trial.suggest_categorical('scale_pos_weight', hps_dict['scale_pos_weight']['values'])
    subsample = trial.suggest_categorical('subsample', hps_dict['subsample']['values'])
    colsample_bytree = trial.suggest_categorical('colsample_bytree', hps_dict['colsample_bytree']['values'])

    model = xgb.XGBClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        learning_rate = learning_rate,
        #scale_pos_weight = scale_pos_weight,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
    )
    
    
    return model

def CNN_optuna(hps_dict, trial):
    # Sequential Layer
    model = Sequential()
    # Covolutional Layer
    model.add(
        Conv2D(
                filters=trial.suggest_categorical("filters", hps_dict['hp_filters']['values']),
                kernel_size=trial.suggest_categorical("kernel_size", hps_dict['hp_kernel_size']['values']),
                strides=trial.suggest_categorical("strides", hps_dict['hp_strides']['values']),
                activation=trial.suggest_categorical("activation", hps_dict['hp_activation']['values']),
                input_shape = (28, 28, 1)
            )
    )
    # Pooling Layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 2)))
    # Flatter the layer to use fully conected layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(data_args_list['num_classes'], activation="softmax"))

    # Compile model with a sampled learning rate.
    optimizer = trial.suggest_categorical("optimizer", hps_dict['optimizer']['values'])
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.02)

    if optimizer == "sgd":
        opt = SGD(learning_rate)
    elif optimizer == "adam":
        opt = Adam(learning_rate)
    elif optimizer == "rmsprop":
        opt = RMSprop(learning_rate)
    
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=opt, metrics=[optuna_args_dict['metric']]
    )
    return model