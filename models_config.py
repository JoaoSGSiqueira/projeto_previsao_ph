from pipeline import main_pipeline
import json
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Flatten
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Permute, Multiply, Flatten, RepeatVector, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge

# You can define any neural network models you want to compare in this dictionary with its respective functions

# Function for Custom Neural Network
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

# Function for LSTM
def build_lstm(input_shape, units, dropout, learning_rate):
    model = Sequential()
    model.add(Input(shape=(input_shape)))  # Define input shape
    model.add(LSTM(units=units[0], return_sequences=True))
    for u in units[1:]:
        model.add(LSTM(units=u, return_sequences=False))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

# Function for GRU
def build_gru(input_shape, units, dropout, learning_rate):
    model = Sequential()
    model.add(Input(shape=(input_shape)))  # Define input shape
    model.add(GRU(units=units[0], return_sequences=True))
    for u in units[1:]:
        model.add(GRU(units=u, return_sequences=False))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

# Function for CNN
def build_cnn(input_shape, filters, kernel_size, dropout, learning_rate):
    model = Sequential()
    model.add(Input(shape=(input_shape)))  # Define input shape
    model.add(Conv1D(filters=filters[0], kernel_size=kernel_size, activation='relu'))
    for f in filters[1:]:
        model.add(Conv1D(filters=f, kernel_size=kernel_size, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

# Function to add attention to LSTM
def build_attention_lstm(input_shape, lstm_units, dropout, learning_rate):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    lstm_out = Dropout(dropout)(lstm_out)
    
    # Attention mechanism
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(lstm_units)(attention)
    attention = Permute([2, 1])(attention)
    
    attended_lstm_out = Multiply()([lstm_out, attention])
    output = Dense(1, activation='linear')(Flatten()(attended_lstm_out))
    
    model = Model(inputs, output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

def build_transformer(input_shape, embed_dim, num_heads, ff_dim, dropout, learning_rate):
    inputs = Input(shape=input_shape)
    
    # Multi-head self-attention layer
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    
    # Feed-forward network
    ffn_output = Dense(ff_dim, activation='relu')(attention_output)
    ffn_output = Dense(input_shape)(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    ffn_output = Add()([attention_output, ffn_output])
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)
    
    # Output layer
    output = Dense(1, activation='linear')(ffn_output)
    
    model = Model(inputs, output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model


regression_models_comparisons = {
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
        "grid_search": True,
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "min_samples_split": [2, 5]
        }
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
        "grid_search": True,
        "param_grid": {
            "hidden_layer_sizes": [(64, 32, 16), (128, 64, 32)],
            "alpha": [0.0001, 0.001],
            "learning_rate": ["constant", "invscaling", "adaptive"]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(),
        "grid_search": True,
        "param_grid": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    },
    "AdaBoost": {
        "model": AdaBoostRegressor(),
        "grid_search": True,
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0]
        }
    },
    "SVR (Support Vector Regressor)": {
        "model": SVR(),
        "grid_search": True,
        "param_grid": {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1, 10],
            "epsilon": [0.01, 0.1, 1.0]
        }
    },
    "Gaussian Process Regressor": {
        "model": GaussianProcessRegressor(),
        "grid_search": False,  # Computationally expensive to optimize
    },
    "Kernel Ridge Regression": {
        "model": KernelRidge(),
        "grid_search": True,
        "param_grid": {
            "alpha": [0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"]
        }
    },
    "LightGBM": {
        "model": LGBMRegressor(),
        "grid_search": True,
        "param_grid": {
            "num_leaves": [31, 50, 100],
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [100, 200],
            "max_depth": [3, 5, -1]  # -1 indicates no limit on depth
        }
    }
}

neural_network_comparisons =  {
    "Custom NN": {
        "model": KerasRegressor(
            model=lambda: build_custom_nn(
                input_dim=6,  # don't change because its the number of features
                layers=[64, 32, 16],
                dropout=0.3,
                learning_rate=0.001
            ),
            epochs=10,
            batch_size=32,
            shuffle=False # don't shuffle the data because it's time series
        ),
        "grid_search": False,
    },
     "Custom RNN LSTM": {
        "model": KerasRegressor(
            model=lambda: build_lstm(
                input_shape=(103, 8),
                units=[64, 32],
                dropout=0.3,
                learning_rate=0.001
            ),
            epochs=10,
            batch_size=32,
            shuffle=False
        ),
        "grid_search": False,
    },"Custom RNN GRU": {
        "model": KerasRegressor(
            model=lambda: build_gru(
                input_shape=(103, 8),
                units=[64, 32],
                dropout=0.3,
                learning_rate=0.001
            ),
            epochs=10,
            batch_size=32,
            shuffle=False
        ),
        "grid_search": False,
    },
    "Custom RNN CNN": {
        "model": KerasRegressor(
            model=lambda: build_cnn(
                input_shape=(103, 8),
                filters=[32, 16],
                kernel_size=3,
                dropout=0.3,
                learning_rate=0.001
            ),
            epochs=10,
            batch_size=32,
            shuffle=False
        ),
        "grid_search": False,
    },
    "Custom RNN Attention LSTM": {
        "model": KerasRegressor(
            model=lambda: build_attention_lstm(
                input_shape=(103, 8),
                lstm_units=64,
                dropout=0.3,
                learning_rate=0.001
            ),
            epochs=10,
            batch_size=32,
            shuffle=False
        ),
        "grid_search": False,
    },
    "Custom NN Transformer": {
        "model": KerasRegressor(
            model=lambda: build_transformer(
                input_shape=(103, 8),
                embed_dim=32,
                num_heads=4,
                ff_dim=32,
                dropout=0.3,
                learning_rate=0.001
            ),
            epochs=10,
            batch_size=32,
            shuffle=False
        ),
        "grid_search": False,
    }
}

transformer = {
    "Custom RNN Attention LSTM": {
    "model": KerasRegressor(
        model=lambda: build_attention_lstm(
            input_shape=(103, 8),  # Specify the sequence length and number of features
            lstm_units=64,
            dropout=0.3,
            learning_rate=0.001
        ),
        epochs=10,
        batch_size=32,
        shuffle=False
    ),
    "grid_search": False,
},
}