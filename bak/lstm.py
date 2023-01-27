from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stock import Stock

### Get Stock data ready
my_stock = Stock('0700.hk', 5, 0.1)
my_stock.get_ready_for_lstm()

### plot
#my_stock.plot()

### Initialising RNN
regressor = Sequential()

### Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (my_stock.X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

### Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

### Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

### Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

### Adding the output layer
regressor.add(Dense(units = 1))

### Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

### Fitting the RNN
regressor.fit(my_stock.X_train, my_stock.y_train, epochs = 3, batch_size = 32)

### plot
predicted_profit = regressor.predict(my_stock.X_test)

plt.plot(my_stock.y_test, color = 'red', label = my_stock.ticker + ' Profit')
plt.plot(predicted_profit, color = 'blue', label = 'Predicted ' + my_stock.ticker +' Profit')
plt.title(my_stock.ticker + ' Stock Profit Prediction')
plt.xlabel('Time')
plt.ylabel(my_stock.ticker + ' Stock Profit')
plt.legend()
plt.show()