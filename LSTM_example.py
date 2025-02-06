import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
tf.random.set_seed(42)

# -------------------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------------------

def convert_to_supervised(data, n_in=1, n_out=1, dropnan=False, replacenan=True):
    """
    data: Dataframe to be used to create a supervised learning set.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    """
    n_features = data.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [('x%d(t-%d)' % (j+1, i)) for j in range(n_features)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data.shift(-i))
        if i == 0:
          names += [('y%d(t)' % (j+1)) for j in range(n_features)]
        else:
          names += [('y%d(t+%d)' % (j+1, i)) for j in range(n_features)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    if replacenan:
        agg = agg.fillna(agg.rolling(7, min_periods=1, center=True).mean())
    return agg


# load the dataset
dataframe = pd.read_csv('')
# Set index as DateTime
dataframe = dataframe.set_index('Month')
# Get all numeric features
dataframe_num = dataframe.select_dtypes(include=['float', 'int'])
numeric_features = [col for col in dataframe_num.columns]
# Make sure all numeric features are float
for feature in numeric_features:
    dataframe = dataframe.astype({feature:'float'})

# User input
history = 12 #past values used by model
future = 6 #future values to predict

# Apply lag to target variables
full_data = convert_to_supervised(dataframe, n_in=history, n_out=future, dropnan=True, replacenan=True)

# Define independent and dependent variables
X_cols = [col for col in full_data.columns if col.startswith('x')]
y_cols = [col for col in full_data.columns if col.startswith('y')]
X = full_data[X_cols].values
y = full_data[y_cols].values

# Create train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

X_train = X_train.reshape(X_train.shape[0], history, 1)
X_test = X_test.reshape(X_test.shape[0], history, 1)



######################### BAYESIAN OPTIMIZATION ###############################

# Get the hypermodel for optimization
from bayes_opt import *

# Define tuner
tuner = kt.BayesianOptimization(
    MyHyperModel(X_train, X_test, history, future),
    objective="val_loss",
    max_trials=100,
    seed = 42,
    overwrite=True)

early_stop = EarlyStopping(patience=4)

# Start the search
tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[early_stop])

# Inspect the search summmary (optional)
tuner.results_summary()

# Get the best model
best_model = tuner.get_best_models()[0]

# Build the model.
best_model.build()
best_model.summary()

# Save/Load the model
model_filename = "best_model.h5"
best_model.save(model_filename)

model = load_model(model_filename)

y_pred = model.predict(X.reshape(X.shape[0], history, 1))


# def build_model(history, future, train_set, test_set):
#     model = Sequential()
#     model.add(tf.keras.Input(shape=(train_set.shape[1], test_set.shape[2])))
#     model.add(tf.keras.layers.Conv1D(filters=6, kernel_size=5, activation='relu'))
#     model.add(LSTM(12, return_sequences=True, activation='relu'))
#     model.add(LSTM(12, return_sequences=False, activation='relu'))
#     model.add(Dense(future))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model

# model_cnnlstm = build_model(history, future, X_train, X_test)
# model_cnnlstm.summary()

hist = model_cnnlstm.fit(X_train, y_train, epochs=1000, batch_size=16, verbose=0, validation_data=(X_test, y_test))

y_pred = model_cnnlstm.predict(X.reshape(X.shape[0], history, 1))

pred_cnn_lstm = []

truth = []

for i in range(len(y_pred)):
    if i==(len(y_pred)-1):
        for j in range(len(y_pred[i])):
            pred_cnn_lstm.append(y_pred[i][j])
            truth.append(y[i][j])    
    else:
        pred_cnn_lstm.append(y_pred[i][0])
        truth.append(y[i][0])


fig = plt.figure(figsize=(30,10))
plt.rcParams.update({'font.size': 18})

months = [i for i in range(len(truth))]

plt.plot(months[0:len(y_train)], truth[0:len(y_train)],label='Train Data', lw=6)
plt.plot(months[len(y_train):], truth[len(y_train):],label='Test Data', lw=6)

plt.plot(pred_cnn_lstm,label='Predictions', lw=4, linestyle='dashed')


plt.vlines(x=len(y_train), ymin=100, ymax = 600, lw=4, linestyle='dashed', color='r')

plt.xlabel('Months', fontsize = 25)
plt.ylabel('Count', fontsize = 25)

plt.legend(fontsize = 30)
plt.show()