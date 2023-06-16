import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.initializers import Constant
from sklearn.metrics import classification_report 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from numpy.random import seed


def run_model(dataframe, path, run, num_timesteps, random_state=None):

    # normalization, i.e., standardization
    scaler_for_X_train = StandardScaler()
    scaler_for_Y_train = StandardScaler()
    scaler_for_X_test = StandardScaler()
    scaler_for_Y_test = StandardScaler()

    data = dataframe

    data.fillna(value=0, inplace=True)
    data.fillna(method='bfill', inplace=True)
    data = data.sort_values(by=['COUNTY', 'HARVESTED', 'MONTH'])
    data.sort_values(by=['HARVESTED', 'COUNTY', 'MONTH'])
    del data['MONTH']
    data = pd.get_dummies(data, columns=['STATE', 'CLIMATE_ZONE'], prefix = ['STATE', 'CLIMATE_ZONE'])

    # model loop
    for i in range(2001, 2021):
        test_data = data[data['HARVESTED'] == i]
        train_data = data[data['HARVESTED'] != i]
        
        test_infos = test_data[['COUNTY', 'CODE', 'SOY_AREA']].copy()
        test_infos.reset_index(inplace=True)
        test_infos = test_infos.drop_duplicates(subset=['COUNTY', 'CODE', 'SOY_AREA'])
        del train_data['COUNTY']; del test_data['COUNTY']
        del train_data['CODE']; del test_data['CODE']
        del train_data['SOY_AREA']; del test_data['SOY_AREA']

        del train_data['HARVESTED']; del test_data['HARVESTED']
        Y_train_all = train_data[['YIELD', 'YIELD_TREND', 'YIELD_TREND_CORRECTED']]
        Y_train_yield = train_data[['YIELD']]
        Y_train_yield_trend = train_data[['YIELD_TREND']]
        Y_train_yied_trend_corrected = train_data[['YIELD_TREND_CORRECTED']]
        X_train = train_data.loc[:, (train_data.columns != 'YIELD') & (train_data.columns != 'YIELD_TREND') & (train_data.columns != 'YIELD_TREND_CORRECTED')]
        Y_test_all = test_data[['YIELD', 'YIELD_TREND', 'YIELD_TREND_CORRECTED']]
        Y_test_yield = test_data[['YIELD']]
        Y_test_yield_trend = test_data[['YIELD_TREND']]
        Y_test_yied_trend_corrected = test_data[['YIELD_TREND_CORRECTED']]
        X_test = test_data.loc[:, (test_data.columns != 'YIELD') & (test_data.columns != 'YIELD_TREND') & (test_data.columns != 'YIELD_TREND_CORRECTED')]
        std_dev = np.std(Y_test_yied_trend_corrected)
        std_dev = np.array(std_dev)
        mean = np.mean(Y_test_yied_trend_corrected)
        mean = np.array(mean)

        X_train_scaled = scaler_for_X_train.fit_transform(X_train)
        X_test_scaled = scaler_for_X_test.fit_transform(X_test)
        
        Y_train_yied_trend_corrected_scaled = scaler_for_Y_train.fit_transform(Y_train_yied_trend_corrected)
        Y_test_yied_trend_corrected_scaled = scaler_for_Y_test.fit_transform(Y_test_yied_trend_corrected)


        num_timesteps = num_timesteps  # Sept, Oct, Nov, Dec, Jan, Feb, March
        num_samples = int(X_train.shape[0] / num_timesteps)  # 7
        num_features = X_train.shape[1]  # 31

        X_train_scaled = X_train_scaled.reshape(num_samples, num_timesteps, num_features) # tuple has to be --> (sample size for each full time step, time step, number of feature)
        Y_train_yied_trend_corrected_scaled = Y_train_yied_trend_corrected_scaled.reshape(num_samples, num_timesteps)

        num_test_samples = int(X_test_scaled.shape[0] / num_timesteps)
        X_test_scaled = X_test_scaled.reshape(num_test_samples, num_timesteps, num_features) # tuple has to be --> (sample size for each full time step, time step, number of feature)
        Y_test_yied_trend_corrected_scaled = Y_test_yied_trend_corrected_scaled.reshape(num_test_samples, num_timesteps)
    
        X_test_scaled = X_test_scaled.reshape(num_test_samples, num_timesteps, num_features) # tuple has to be --> (sample size for each full time step, time step, number of feature)
        Y_test_yied_trend_corrected_scaled = Y_test_yied_trend_corrected_scaled.reshape(num_test_samples, num_timesteps)
        
        # model
        model = Sequential()
        model.add(LSTM(128, input_shape=(num_timesteps, num_features), return_sequences = True)) # 10 lstm neuron(block)
        model.add(Dropout(0.25))
        model.add(LSTM(64))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        history = model.fit(X_train_scaled, Y_train_yied_trend_corrected_scaled, epochs=100, shuffle=False, verbose = 0,
                validation_split=0.2 ,
                batch_size=133, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
        predictions = model.predict(X_test_scaled)
        predictions = scaler_for_Y_test.inverse_transform(predictions)
        Y_test_yield_trend = np.array(Y_test_yield_trend)
        yield_trend = []

        y_true_test_length = Y_test_yield.shape[0]  # the 2d time series matrix of num_test_samples * num_timesteps
        
        for z in range(0,y_true_test_length,num_timesteps):
            yield_trend.append(Y_test_yield_trend[z])
            
        yield_trend = np.array(yield_trend)
        predictions = predictions + yield_trend
        Y_test_yield = np.array(Y_test_yield)
        test_set = []
        
        for p in range(0,y_true_test_length,num_timesteps):
            test_set.append(Y_test_yield[p])
            
        test_set = np.array(test_set)
        test_set = test_set.reshape(-1,1)    

        predictions_ = predictions.reshape((predictions.shape[0],))
        y_true__ = np.array(test_set)
        y_true__ = y_true__.reshape((y_true__.shape[0],))
        index = np.arange(0, predictions_.shape[0])
        
        df = {'Predicted' : predictions_, 'Observed' : y_true__}
        save_predictions = pd.DataFrame(df, index=index)
        save_predictions.reset_index(inplace=True); test_infos.reset_index(inplace=True)  # required to achieve a correct matching of results and information
        results = pd.concat([save_predictions, test_infos], axis=1)
        del results['index']; del results['level_0']
        
        # calculate the production
        results['Observed_production'] = results['Observed'] * results['SOY_AREA']
        results['Predicted_production'] = results['Predicted'] * results['SOY_AREA']   

        results.to_csv('{}/{}_{}_LSTM.csv'.format(path, run, i), index=False)

        

