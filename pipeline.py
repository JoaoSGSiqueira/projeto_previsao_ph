import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import TimeSeriesSplit

# Função para carregar e preparar os dados
def prepare_data(df, target_column, test_size=0.2, sample_size=None):
    """
    Prepares time series data by ensuring newer data is used for testing.

    Parameters:
        df (pd.DataFrame): The dataset.
        target_column (str): The target variable column name.
        test_size (float): Proportion of data to use for testing.
        sample_size (int, optional): Number of samples to randomly select from the dataset.

    Returns:
        X_train, X_test, y_train, y_test: Splits of the dataset.
    """
    # Drop unused columns and handle missing values
    df = df.drop(['Real_RD_MV_ValvulaCalpHDosado-2'], axis=1)
    df = df.dropna()
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Split into features and target
    X = df.drop(columns=[target_column, 'ID', 'timezone'])
    y = df[target_column]
    
    # Calculate the split index for time series
    split_index = int(len(df) * (1 - test_size))
    
    # Split into train and test sets
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    return X_train, X_test, y_train, y_test


# Função para treinar e avaliar modelos tradicionais
def train_ml_model(model, X_train, y_train, X_test, y_test, grid_search=False, param_grid=None):
    if grid_search and param_grid:
        tscv = TimeSeriesSplit(n_splits=5)
        model = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
        model.fit(X_train, y_train)
        best_params = model.best_params_
        model = model.best_estimator_
        print(f"Best parameters for {model}: {best_params}")
    else:
        model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Log parameters if available
    params = model.get_params() if hasattr(model, "get_params") else "Not available"
    
    return {"rmse": rmse, "mae": mae, "r2": r2, "params": params}, model

# Função para criar e treinar redes neurais personalizadas
def train_custom_nn(X_train, X_test, y_train, y_test, config):
    layers = config.get('layers', [64, 32])
    dropout = config.get('dropout', 0.2)
    learning_rate = config.get('learning_rate', 0.001)
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 32)
    
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', input_dim=X_train.shape[1]))
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)
    
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Log NN config
    params = {
        "layers": layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size
    }
    
    return {"rmse": rmse, "mae": mae, "r2": r2, "params": params}, model

# Função principal para pipeline
def main_pipeline(dataset_path, target_column, models_config, sample_size=None):
    df = pd.read_csv(dataset_path)
    X_train, X_test, y_train, y_test = prepare_data(df, target_column, sample_size=sample_size)

    results = []
    trained_models = {}
    
    for name, config in models_config.items():
        print(f"\nTraining model: {name}")
        model = config.get('model')
        optimized = config.get('grid_search', False)
        param_grid = config.get('param_grid', None)
        
        if name.startswith("Custom NN"):
            # Train a custom neural network
            metrics, trained_model = train_custom_nn(X_train.values, X_test.values, y_train.values, y_test.values, config)
        else:
            # Train a traditional model
            metrics, trained_model = train_ml_model(model, X_train, y_train, X_test, y_test, optimized, param_grid)
        
        results.append({"model_name": name, **metrics})
        trained_models[name] = trained_model
        print(f"{name} Metrics: RMSE={metrics['rmse']}, MAE={metrics['mae']}, R2={metrics['r2']}")

    # Sort results by RMSE
    results_df = pd.DataFrame(results).sort_values(by="rmse")
    best_model_name = results_df.iloc[0]['model_name']
    print(f"\nBest model: {best_model_name} with RMSE = {results_df.iloc[0]['rmse']}")

    return results_df, trained_models[best_model_name]


def build_custom_nn(input_dim, layers, dropout, learning_rate):
    """
    Builds a custom neural network with the specified parameters.

    Parameters:
        input_dim (int): Number of input features.
        layers (list of int): Number of neurons in each hidden layer.
        dropout (float): Dropout rate (between 0 and 1).
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        keras.Model: Compiled Keras model.
    """

    model = Sequential()
    # Input layer
    model.add(Dense(layers[0], activation='relu', input_dim=input_dim))

    # Hidden layers
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    
    return model


# Exemplo de JSON-like models_config
models_config = {
    "Decision Tree": {
        "model": DecisionTreeRegressor(),
        "grid_search": True,
        "param_grid": {
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5, 10]
        }
    },
    "Random Forest": {
        "model": RandomForestRegressor(),
        "optimized": False
    },
    "XGBoost": {
        "model": XGBRegressor(),
        "grid_search": True,
        "param_grid": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    },
    "MLP Regressor": {
        "model": MLPRegressor(max_iter=500),
        "grid_search": False
    }
}

custom =  {"Custom NN": {
        "model": KerasRegressor(
            build_fn=lambda: build_custom_nn(
                input_dim=10,  # Replace with the actual number of input features
                layers=[64, 32, 16],
                dropout=0.3,
                learning_rate=0.001
            ),
            epochs=100,
            batch_size=64,
            verbose=1
        ),
        "optimized": False  # Set to True if you want to run hyperparameter tuning
    }}

# Exemplo de execução
results_df, best_model = main_pipeline("Dados Brutos pHDosado - Completo.csv", "Real_RD_PV_pHDosado", models_config, sample_size=10000)
results_df.to_csv("results.csv", index=False)