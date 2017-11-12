# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 00:42:18 2017

@author: buddy
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:59:12 2017

@author: buddy
"""
import matplotlib.pyplot as plt
        
import numpy as np
import pandas as pd

time_steps = 60

traning_set_init = pd.read_csv('avg_BTC_price_USD.csv', sep='\t',encoding='utf-8')
#do testow
traning_presplit = traning_set_init.ix[:,5:6]
#traning_presplit = traning_set_init.iloc[:, 1:2].values
traning_presplit = traning_presplit.dropna(axis=0)

train_ab = traning_presplit.iloc[:2230]
#validate_ab = traning_presplit.iloc[2000:2116]
test_ab = traning_presplit.iloc[2230:len(traning_set_init)]

time_predicting = len(train_ab) #days in traning data
test_size = len(test_ab) #days to predict in test data

dataset_test_real_stock_price=test_ab

#traning_set = np.asarray(traning_set)
#traning_set = [x for x in traning_set if str(x) != 'nan']
#spliting data into train validate and test set
#train, validate, test = np.split(traning_set, [int(.8 * len(traning_set)), int(.9 * len(traning_set))])

#train, validate, test = np.split(traning_set, [int(.8 * len(traning_set)), int(.9 * len(traning_set))])

#scaler.fit expects 2d matrix so...
#train_ab_scaled = np.reshape(train_ab['avg_btc_price_usd'], (-1, 1))
#validate = np.reshape(validate, (-1, 1))
#test = np.reshape(test, (-1, 1))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

train_scaled = scaler.fit_transform(train_ab)


X_train = []
y_train = []
for i in range(time_steps, len(train_ab)):
    X_train.append(train_scaled[i-time_steps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


##getting real price
#dataset_test = test
#test_set  = np.reshape(dataset_test, (-1, 1))
#real_stock_price = np.concatenate((train[0:len(train)], test_set), axis = 0)


dataset_test = dataset_test_real_stock_price
real_stock_price = dataset_test_real_stock_price
#real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
#scaled_real_stock_price = scaler.fit_transform(real_stock_price)

#dataset_total = np.concatenate((train, data_con), axis = 0)
dataset_total = pd.concat((train_ab['avg_btc_price_usd'], dataset_test['avg_btc_price_usd']), axis = 0)

#inputs = dataset_total[len(dataset_total) - len(dataset_test) - time_predicting:].values
#inputs = inputs.reshape(-1,1)
#inputs = scaler.transform(inputs)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - time_steps:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test = []

for i in range(time_steps, time_steps + test_size):
    X_test.append(inputs[i-time_steps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


real_price_len = np.arange(len(real_stock_price))
# Visualising the results
plt.figure('BTC price prediction NEW dropout 50 units')
plt.plot(real_price_len,real_stock_price, color = 'red', label = 'Real BTC Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted BTC Price')
plt.title('BTC Price Prediction')
plt.xlabel('Time')
plt.ylabel('BTC Price')
plt.grid(True)
plt.legend()
plt.show()



plt.figure('BTC testdata')
plt.plot(train_ab)
plt.show()
