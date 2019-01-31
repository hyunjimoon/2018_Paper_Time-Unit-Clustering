import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping


class NextoptLSTM(object):
    def __init__(self, early_stop=True, patience=3):
        self.model = None
        self.hist = None

        if early_stop:
            self.early_stop = EarlyStopping(monitor='loss', patience=patience)

    def fit(self, X_train=None, y_train=None, X_val=None, y_val=None, lags=7, epochs=100, batch_size=1, verbose=2):
        self.model = Sequential()
        self.model.add(Dense(3, input_shape=(1, lags), activation='relu'))
        self.model.add(LSTM(6, activation='relu'))
        self.model.add(Dense(1, activation='relu'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.hist = self.model.fit(X_train, y_train,
                                   epochs=epochs,
                                   batch_size=batch_size,
                                   verbose=verbose,
                                   callbacks=[self.early_stop])

    def predict(self, X_train, X_test):
        fitted_value = self.model.predict(X_train)
        predicted_value = self.model.predict(X_test)

        return fitted_value, predicted_value

    def plot_history(self):
        fig, loss_ax = plt.subplots()

        acc_ax = loss_ax.twinx()

        loss_ax.plot(self.hist.history['loss'], 'y', label='train loss')
        loss_ax.plot(self.hist.history['val_loss'], 'r', label='val loss')

        acc_ax.plot(self.hist.history['acc'], 'b', label='train acc')
        acc_ax.plot(self.hist.history['val_acc'], 'g', label='val acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')

        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')

        plt.show()


