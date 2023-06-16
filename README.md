# ML_Soybean_Yield_Forecasting_Brazil
The following data and codes are related to the paper "Machine learning for soybean yield forecasting in Brazil" for reproducible results at the municipality level.

## Content

`Data_cast_Paper.csv`: The data used for the models except LSTM. It consists of 24860 rows and 128 columns. <br>
`LSTM_data.csv`: The data used for the Long Short Term Memory **(LSTM)** model. It consists of 174020 rows and 28 columns. <br>
`requirements.txt`: The main library requirements with corresponding versions. <br>
`random_states.txt`: Definition. <br>
`ANN_forecasting.py`: The main code script for vanilla Artificial Neural Network **(ANN)** model. The model structure and hyperparameters as follows respectively. <br><br>
<img src= "https://github.com/maltevb/ML_Soybean_Yield_Forecasting_Brazil/assets/63941775/f3a8591d-fb48-4664-b4b5-a5fea6e9ffaf" width=500>

| **Hyperparameter** | **Unit** |
| :---:        |    :---:   |
| Loss:      | MSE |
| Optimizer:   | Adam |
| Activation:   | ReLu |
| Learning rate:   | $$10^{-4}$$ |
| Batch size:   | $$32$$ |
| Metric:   | Validation loss |
| Early stopping:   | $$10$$ epochs |
| Validation split:   | $$20 %$$ |
| Dropout rate:   | $$30 %$$ for each hidden layer |

`LSTM_forecasting.py`: The main code script for Long Short Term Memory **(LSTM)** model. The model structure and hyperparameters as follows respectively. <br><br>
<img src= "https://github.com/maltevb/ML_Soybean_Yield_Forecasting_Brazil/assets/63941775/a360c3a4-d2c1-493a-bc8a-e3d647193ed1" width=500>

| **Hyperparameter** | **Unit** |
| :---:        |    :---:   |
| Loss:      | MSE |
| Optimizer:   | Adam |
| Learning rate:   | $$10^{-4}$$ |
| Batch size:   | $$19$$ |
| Metric:   | Validation loss |
| Early stopping:   | $$10$$ epochs |
| Validation split:   | $$20 %$$ |
| Dropout rate:   | $$20 %$$ for each hidden layer |

`RF_forecasting.py`: The main code script for Random Forest model. The hyperparameters as follows. <br>

| **Hyperparameter** | **Unit** |
| :---:        |    :---:   |
| Number of trees  | $$100$$ (default) |

`Ridge_forecasting.py`: The main code script for Ridge regression model. The hyperparameters as follows. <br>

| **Hyperparameter** | **Unit** |
| :---:        |    :---:   |
| &alpha; (L2 regularization coefficient)  | $$1.0$$ (default) |

`XGBoost_forecasting.py`: The main code script for XGBoost model. The hyperparameters as follows. <br>

| **Hyperparameter** | **Unit** |
| :---:        |    :---:   |
| Maximum depth   | $$4$$ |
| Number of estimators   | $$70$$ |
| Learning rate   | $$0.1$$ |
| &alpha;   | $$3$$  |
| &gamma;   | $$4.5$$  |
| &lambda;   | $$1.5$$  |

`main.py`: The main code script for seasonal yield estimation.



