# Generalized Forecasting Implementation

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor  # Example: another model option
plt.style.use('fivethirtyeight')

# Function to load data
def get_data(folder, data_name, delimiter=';', decimal=','):
    file = folder / data_name
    data = pd.read_csv(file, delimiter=delimiter, decimal=decimal)
    return data

# Generic function to add external data (e.g., wind, solar, temperature, etc.)
def add_external_data(feature_name, file_name, dataframe, folder):
    '''
    Adds additional data (e.g., weather data) to the main dataset.
    feature_name: name of the feature to be added (e.g., 'SOLAR', 'WIND', 'TEMP')
    file_name: the file from which the additional data is loaded
    dataframe: main dataframe where the additional data will be added
    folder: path of the folder where the data file is located
    '''
    data = get_data(folder, file_name)
    data = pd.DataFrame(data)
    data, _ = data_preprocessing(data)
    dataframe[feature_name] = data.iloc[:, 0] * -1
    return dataframe

# Data preprocessing function
def data_preprocessing(dataframe):
    '''
    Prepares the dataframe for the forecast.
    dataframe: dataframe to be preprocessed
    '''
    dataframe.columns = [f'Var_{i}' if i >= 2 else name for i, name in enumerate(dataframe.columns)]
    dataframe['DateTime'] = pd.to_datetime(dataframe['Var_0'] + "-" + dataframe['Var_1'], format='%d/%m/%Y-%H:%M')
    dataframe = dataframe.drop(['Var_0', 'Var_1'], axis=1).set_index('DateTime')
    
    # Handling NaNs
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    dataframe[numeric_cols] = dataframe[numeric_cols].fillna(dataframe.rolling(5, min_periods=1, center=True).mean())
    return dataframe, numeric_cols

# Train-test split function
def divide_data_to_train_test(dataframe, horizon_days):
    """
    Splits the data into train and test sets.
    horizon_days: Number of days in the past to use for training
    """
    total_days = dataframe.shape[0] // 24
    if horizon_days > total_days:
        raise Exception("Horizon exceeds the number of days in the dataset.")
    else:
        end_train = horizon_days * 24
        data_train = dataframe.iloc[-end_train:-24]
        data_test = dataframe.iloc[-24:]
        
        # Add time-related features
        for df in [data_train, data_test]:
            df['Hour'] = df.index.hour
            df['Week Day'] = df.index.weekday
    return data_train, data_test

# Create train-test sets
def create_train_test_sets(data_train, data_test, additional_features=None):
    """
    additional_features: list of additional feature names (e.g., ['SOLAR', 'WIND'])
    """
    features = ['Hour', 'Week Day'] + (additional_features or [])
    X_train = data_train[features]
    y_train = data_train.drop(features, axis=1)
    X_test = data_test[features]
    y_test = data_test.drop(features, axis=1)

    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)

# Function to convert data into supervised learning format (lagging)
def convert_to_supervised(data, n_in=1, n_out=1, dropnan=False):
    """
    data: Dataframe to be used to create a supervised learning set.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    """
    n_features = data.shape[1]
    cols, names = list(), list()

    # Input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_features)]

    # Forecast sequence (t, t+1, ..., t+n)
    for i in range(n_out):
        cols.append(data.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_features)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_features)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # Drop NaNs
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Generic model tuning function
def tune_model(X_train, y_train, model_type='xgboost', param_grid=None):
    """
    Generalized model tuning function.
    model_type: 'xgboost', 'random_forest', etc.
    param_grid: dictionary of hyperparameters
    """
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(objective="reg:squarederror", tree_method='hist')
        default_params = {
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'subsample': np.arange(0.5, 1.0, 0.1),
            'colsample_bytree': np.arange(0.5, 1.0, 0.1),
            'n_estimators': [100, 200, 300, 400, 500]
        }
    elif model_type == 'random_forest':
        model = RandomForestRegressor()
        default_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    param_grid = param_grid or default_params
    search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=50, scoring='neg_root_mean_squared_error', n_jobs=3, verbose=1)
    search.fit(X_train, y_train)
    return search.best_estimator_

# Function to plot results
def plot_predictions(X_test, y_test, predictions, comparison_data=None, title="Predictions vs Observations"):
    plt.figure(figsize=(10, 6))
    plt.plot(X_test['Hour'], predictions, label='Predictions', color='red')
    plt.plot(X_test['Hour'], y_test, label='Actual', color='green')
    if comparison_data is not None:
        plt.plot(X_test['Hour'], comparison_data, label='Comparison', color='blue', linestyle='--')
    plt.title(title)
    plt.xlabel("Hour")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # User configuration
    config = {
        'horizon': 22,
        'add_solar': True,
        'add_wind': True,
        'folder_path': r"C:\path_to_data",
        'main_data': "data_electricity.csv",
        'forecast_data': "ebase_electricity.csv",
        'solar_data': "solar_data.csv",
        'wind_data': "wind_data.csv",
        'model_type': 'xgboost'  # Can be 'random_forest' or other
    }

    # Load data
    folder = Path(config['folder_path'])
    main_data = get_data(folder, config['main_data'])
    main_data, features = data_preprocessing(main_data)

    # Add additional data if specified
    if config['add_solar']:
        main_data = add_external_data('SOLAR', config['solar_data'], main_data, folder)
    if config['add_wind']:
        main_data = add_external_data('WIND', config['wind_data'], main_data, folder)

    # Train-test split
    data_train, data_test = divide_data_to_train_test(main_data, config['horizon'])

    # Create train-test sets
    X_train, X_test, y_train, y_test = create_train_test_sets(data_train, data_test, additional_features=['SOLAR', 'WIND'] if config['add_solar'] and config['add_wind'] else None)

    # Model tuning
    model = tune_model(X_train, y_train, model_type=config['model_type'])

    # Predict
    predictions = model.predict(X_test)

    # Plot results
    plot_predictions(X_test, y_test, predictions)
