import numpy as np 
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.initializers import Constant
from sklearn.preprocessing import StandardScaler


def run_model(dataframe, path, run, random_state=None):

    # normalization, i.e., standardization
    scaler_for_X_train = StandardScaler()
    scaler_for_Y_train = StandardScaler()
    scaler_for_X_test = StandardScaler()
    scaler_for_Y_test = StandardScaler()

    data = dataframe.copy()
    data = data.dropna(subset=['SOY_AREA'])

    # the workaround of the .T is that the fillna method just works with with row-wise, but not column wise -> this would be needed 
    # as the we deal with time-series data and the value is in the column before.
    data = data.T
    data.fillna(method='ffill', inplace=True)
    data = data.T
    data = data.sort_values(by=['COUNTY','HARVESTED'])
    data = data.reset_index(drop=True)
    data['CLIMATE_ZONE'].value_counts()/20
    data['STATE'].value_counts()/20

    # one hot encoding
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
        Y_test_yied_trend_corrected = test_data[['YIELD_TREND_CORRECTED']]
        X_test = test_data.loc[:, (test_data.columns != 'YIELD') & (test_data.columns != 'YIELD_TREND') & (test_data.columns != 'YIELD_TREND_CORRECTED')]

        scaler_for_Y_train.fit(Y_train_yied_trend_corrected)
        scaler_for_X_train.fit(X_train)
        scaler_for_Y_test.fit(Y_test_yied_trend_corrected)
        scaler_for_X_test.fit(X_test)
        X_test_scaled = scaler_for_X_test.transform(X_test)

        # model
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(133,)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))
        adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss='mean_squared_error', optimizer=adam)

        predictions = model.predict(X_test_scaled)

        predictions = scaler_for_Y_test.inverse_transform(predictions)
        Y_test_yield_trend = np.array(Y_test_yield_trend)
        predictions = predictions + Y_test_yield_trend
        
        predictions_ = predictions.reshape((predictions.shape[0],))
        y_true__ = np.array(Y_test_yield)
        y_true__ = y_true__.reshape((y_true__.shape[0],))
        index = np.arange(0, predictions_.shape[0])
        
        df = {'Predicted' : predictions_, 'Observed' : y_true__}
        save_predictions = pd.DataFrame(df, index=index)
        results = pd.concat([save_predictions, test_infos], axis=1)
        del results['index']
        
        # calculate the production
        results['Observed_production'] = results['Observed'] * results['SOY_AREA']
        results['Predicted_production'] = results['Predicted'] * results['SOY_AREA']
        
        results.to_csv('{}/{}_{}_ANN.csv'.format(path, run, i), index=False)
