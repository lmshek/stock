from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stock import Stock

### Get Stock data ready
my_stock = Stock('0700.hk', 5, 0.1)
my_stock.get_ready_for_ann()

### plot
#my_stock.plot()

### Initialising ANN
regressor = Sequential()

### Adding the input layer and the first hidden layer
regressor.add(Dense(units = 4, input_shape= (2,), activation='relu'))

### Adding the second hidden layer
regressor.add(Dense(units=4, activation = 'relu'))

### Adding the output layer
regressor.add(Activation('softmax'))

### Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mse')

### Fitting the RNN
regressor.fit(my_stock.X_train, my_stock.y_train, epochs = 30, batch_size = 32)

### plot
predicted_profit = regressor.predict(my_stock.X_test)

plt.plot(my_stock.y_test, color = 'red', label = my_stock.ticker + ' Profit')
plt.plot(predicted_profit, color = 'blue', label = 'Predicted ' + my_stock.ticker +' Profit')
plt.title(my_stock.ticker + ' Stock Profit Prediction')
plt.xlabel('Time')
plt.ylabel(my_stock.ticker + ' Stock Profit')
plt.legend()
plt.show()