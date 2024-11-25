from pipeline import main_pipeline
import json
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


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