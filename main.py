import random
from re import M
import pandas as pd
from glob import glob
import sys
import os
import pathlib
import ANN_forecasting
import RF_Forecasting
import Ridge_forecasting
import BiLSTM_forecasting
import LSTM_forecasting
import numpy as np
import random
import tensorflow as tf
import logging
import XGBoost_Forecasting

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# set the random state before the rest of the script runs
run = int(sys.argv[1])
states = np.loadtxt('random_states.txt')
state = int(states[run])
#call all this three to set a random state to generate reproducible random states
np.random.seed(state)
random.seed(state)
tf.random.set_seed(state)

# set the logging file
logging.basicConfig(filename='Training.log', level=logging.WARNING)

TIME_SERIES_MODEL =  True # e.g. LSTM, BiLSTM
IID_MODEL =  True # IID Model are models that do not consider time series 

CURRENT_PATH = pathlib.Path(__file__).parent.resolve()
RESULTS_PATH = '{}/Results/'.format(CURRENT_PATH)


constant_features = ['CODE', 'CLIMATE_ZONE', 'STATE', 'COUNTY', 'HARVESTED', 'YIELD', 'YIELD_TREND', 'YIELD_TREND_CORRECTED', 'SOY_AREA']
series_features = ['MEAN_SRAD', 'MEAN_TMIN', 'MEAN_TMAX', 'HOT_DAYS', 'ACC_RAINFALL', 'LOWRAIN_DAYS', 'MEAN_NDVI', 'MEAN_EVI', 'MEAN_CVI', 'MEAN_GLI', 'SPI1', 
            'SPI3', 'STI1', 'EXC', 'DEF', 'ETP', 'ONI']


def get_idd_features(month_ids):
    features_to_keep = constant_features.copy()
    for i in month_ids:
        for feature in series_features:
            features_to_keep.append('{}_{}'.format(i, feature))


def get_ts_features(data, month_ids):
    data.set_index('MONTH', inplace=True)
    data = data.loc[month_ids]
    data = data.reset_index()
    return data
    
    

# the monthes have different indices
monthes = {'September' : 9,
            'October' : 10,
            'November' : 11,
            'December' : 12,
            'January' : 1,
            'February' : 2,
            'March' : 3}

month_to_keep = []
for month in monthes.keys():
    try:
        month_to_keep.append(monthes[month])  # appending the month integer

        if os.path.isdir('{}/{}/'.format(RESULTS_PATH, month)):
            pass
        else:
            os.mkdir('{}/{}/'.format(RESULTS_PATH, month))

        if IID_MODEL:
            # just continue with the constant features and the time series features of the monthes
            features = get_idd_features(month_to_keep)
            data = pd.read_csv('Data_cast_Paper.csv', usecols=features)

            # apply the ANN 
            path = '{}/{}/0_{}'.format(RESULTS_PATH, month, 'ANN')
            if not os.path.isdir(path):
                os.mkdir(path)
            results = ANN_forecasting.run_model(data, path, run, random_state=None)

            # apply the RF
            path = '{}/{}/0_{}'.format(RESULTS_PATH, month, 'RF')
            if not os.path.isdir(path):
                os.mkdir(path)
            results = RF_Forecasting.run_model(data, path, run)

            # apply the LR (Ridge Regression)
            path = '{}/{}/0_{}'.format(RESULTS_PATH, month, 'LR')
            if not os.path.isdir(path):
                os.mkdir(path)
            results = Ridge_forecasting.run_model(data, path, run)

            # apply the XGBoost
            path = '{}/{}/0_{}'.format(RESULTS_PATH, month, 'XGB')
            if not os.path.isdir(path):
                os.mkdir(path)
            results = XGBoost_Forecasting.run_model(data, path, run)



        if TIME_SERIES_MODEL:
            features = constant_features + series_features
            features.append('MONTH')
            data = pd.read_csv('LSTM_data.csv', usecols=features)
            data= get_ts_features(data, month_to_keep)


            # apply the LSTM
            path = '{}/{}/0_{}'.format(RESULTS_PATH, month, 'LSTM')
            if not os.path.isdir(path):
                os.mkdir(path)
            LSTM_forecasting.run_model(data, path, run,  num_timesteps=len(month_to_keep), random_state=None)

            # apply the BiLSTM
            path = '{}/{}/0_{}'.format(RESULTS_PATH, month, 'BiLSTM')
            if not os.path.isdir(path):
                os.mkdir(path)
            BiLSTM_forecasting.run_model(data, path, run, num_timesteps=len(month_to_keep), random_state=None)
    except Exception as e:
        logging.exception(e, exc_info=True, )  # log exception info at CRITICAL log level




