import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report 
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from numpy.random import seed
from sklearn.ensemble import RandomForestRegressor
seed(1)


def run_model(dataframe, path, run):

    data = dataframe.copy()
    data = data.dropna(subset=['SOY_AREA'])

    # the workaround of the .T is that the fillna method just works with with row-wise, but not column wise -> this would be needed 
    # as the we deal with time-series data and the value is in the column before.
    data = data.T
    data.fillna(method='ffill', inplace=True)
    data = data.T

    data = data.sort_values(by=['COUNTY','HARVESTED'])
    for i in range(10):
        data = data.sample(frac=1)

    data = data.reset_index(drop=True)
    data = pd.get_dummies(data, columns=['STATE', 'CLIMATE_ZONE'], prefix = ['STATE', 'CLIMATE_ZONE'])

    # model loop
    for i in range(2001, 2021):
        test_data = data[data['HARVESTED'] == i]
        train_data = data[data['HARVESTED'] != i]
        
        test_infos = test_data[['COUNTY', 'CODE', 'SOY_AREA']].copy()
        test_infos.reset_index(inplace=True)
        del train_data['COUNTY']; del test_data['COUNTY']
        del train_data['CODE']; del test_data['CODE']
        del train_data['SOY_AREA']; del test_data['SOY_AREA']

        del train_data['HARVESTED']
        del test_data['HARVESTED']
        Y_train_yied_trend_corrected = train_data[['YIELD_TREND_CORRECTED']]
            
        X_train = train_data.loc[:, (train_data.columns != 'YIELD') & (train_data.columns != 'YIELD_TREND') & (train_data.columns != 'YIELD_TREND_CORRECTED')]
            
        Y_test_yield = test_data[['YIELD']]
        Y_test_yield_trend = test_data[['YIELD_TREND']]
        X_test = test_data.loc[:, (test_data.columns != 'YIELD') & (test_data.columns != 'YIELD_TREND') & (test_data.columns != 'YIELD_TREND_CORRECTED')]
       
        ridge = Ridge()
        ridge.fit(X_train, Y_train_yied_trend_corrected)
        predictions = ridge.predict(X_test)
        
        Y_test_yield_trend = np.array(Y_test_yield_trend)
        predictions = np.add(predictions.flatten(), Y_test_yield_trend.flatten())  # add the trend again
        

        y_true__ = np.array(Y_test_yield)
        y_true__ = y_true__.reshape((y_true__.shape[0],))
        index = np.arange(0, predictions.shape[0])

        df = {'Predicted' : predictions, 'Observed' : y_true__}
        save_predictions = pd.DataFrame(df, index=index)
        results = pd.concat([save_predictions, test_infos], axis=1)
        del results['index']
        
        # calculate the production
        results['Observed_production'] = results['Observed'] * results['SOY_AREA']
        results['Predicted_production'] = results['Predicted'] * results['SOY_AREA']
        
        results.to_csv('{}/{}_{}_LR.csv'.format(path, run, i), index=False)

        # check the success of each model loop
        print('Ridge done: {}'.format(i))
