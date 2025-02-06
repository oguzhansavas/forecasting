import tensorflow
from tensorflow import keras
from keras import Input
from keras.models import Sequential
from keras.layers import Input, Conv1D, LSTM, Dense
import keras_tuner as kt
from keras.optimizers import Adam

class MyHyperModel(kt.HyperModel):

    def __init__(self, train_set, test_set, history, future):
        self.train_set = train_set
        self.test_set = test_set
        self.history = history
        self.future = future

    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(self.train_set.shape[1], self.test_set.shape[2])))
        filter = hp.Int("filters", min_value=6, max_value=12, )
        model.add(Conv1D(filters=filter, kernel_size=5, activation='relu'))
        unit = hp.Int("units", min_value=6, max_value=18, )
        model.add(LSTM(unit, return_sequences=True, activation='relu'))
        unit = hp.Int("units", min_value=6, max_value=18, )
        model.add(LSTM(unit, return_sequences=False, activation='relu'))
        model.add(Dense(self.future))
        learn_rate = hp.Float("lr", min_value=1e-4, max_value=1e-1, sampling="log")
        model.compile(optimizer = Adam(learning_rate = learn_rate),
                      loss = "mean_squared_error",
                      metrics = ["mean_squared_error"])
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args,
                         batch_size = hp.Choice("batch_size", [16, 32]),
                         **kwargs)