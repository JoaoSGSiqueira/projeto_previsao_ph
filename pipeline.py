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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

def make_dataset_stationary(df, columns):
    """
    Makes non-stationary time series data stationary by differencing.
    
    Parameters:
        df (pd.DataFrame): The dataset.
        columns (list): List of columns to be checked and made stationary.
        
    Returns:
        pd.DataFrame: The modified dataframe with stationary columns.
    """
    for col in columns:
        # Apply differencing if the series is non-stationary (based on KPSS test)
        if col in ['Real_RD_MV_ValvulaCalpHDosado', 'Real_RD_PV_VazaoDosado', 
                   'Real_RD_MV_PressaoLinhaCal', 'Real_RD_PV_NivelTqCal', 
                   'Real_RD_PV_NivelTqDosado']:
            df[col] = df[col].diff().dropna()
            print(f"Applied differencing to {col}")
        # Optionally, apply log transformation before differencing if data shows exponential growth
        # if some_condition:  
        #     df[col] = np.log(df[col])
        
    return df

def scale_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales all the numerical columns of a DataFrame using StandardScaler.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to scale.
    
    Returns:
    pd.DataFrame: A new DataFrame with scaled values.
    """
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Apply scaling to the numerical columns only
    df_scaled = df.copy()  # Avoid modifying the original DataFrame
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df_scaled

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

    # Ensure timezone column is in datetime format
    df['timezone'] = pd.to_datetime(df['timezone'])
    
    # Sort by the timezone column
    df = df.sort_values(by='timezone')

    if sample_size:
        df = df.iloc[:sample_size]

    # Transfrom the dataset to a stationary one
    df = make_dataset_stationary(df, df.columns)

    # Transfrom the dataset Scale
    df = scale_dataframe(df)

    # remove NaN values
    df = df.dropna()
    
    # Split into features and target
    X = df.drop(columns=[target_column, 'ID', 'timezone'])
    y = df[target_column]
    
    # Calculate the split index for time series
    split_index = int(len(df) * (1 - test_size))
    
    # Split into train and test sets
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train, X_test, y_test, grid_search=False, param_grid=None, cv_splits=5):
    """
    Trains and evaluates a machine learning model, with optional cross-validation and grid search.

    Parameters:
        model: The machine learning model to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        grid_search (bool): Whether to use GridSearchCV for hyperparameter tuning.
        param_grid (dict): Grid of parameters for GridSearchCV.
        cv_splits (int): Number of splits for cross-validation.

    Returns:
        dict: Metrics (RMSE, MAE, R2) and model parameters.
        model: Trained model.
    """
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    cv_rmse_scores, cv_mae_scores, cv_r2_scores = [], [], []
    
    if grid_search and param_grid:
        # Perform grid search with time series split
        model = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Retrieve the best model and parameters
        best_params = model.best_params_
        model = model.best_estimator_
        print(f"Best parameters for {model}: {best_params}")
    
    # Perform manual cross-validation to compute RMSE, MAE, and R²
    for train_index, val_index in tscv.split(X_train):
        X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # Train the model on the current fold
        model.fit(X_train_cv, y_train_cv)
        
        # Predict on the validation set
        y_val_pred = model.predict(X_val_cv)
        
        # Calculate metrics
        cv_rmse_scores.append(np.sqrt(mean_squared_error(y_val_cv, y_val_pred)))
        cv_mae_scores.append(mean_absolute_error(y_val_cv, y_val_pred))
        cv_r2_scores.append(r2_score(y_val_cv, y_val_pred))
    
    # Output cross-validation results
    print(f"Cross-Validation RMSE scores: {cv_rmse_scores}")
    print(f"Average CV RMSE: {np.mean(cv_rmse_scores):.4f}")
    print(f"Cross-Validation MAE scores: {cv_mae_scores}")
    print(f"Average CV MAE: {np.mean(cv_mae_scores):.4f}")
    print(f"Cross-Validation R² scores: {cv_r2_scores}")
    print(f"Average CV R²: {np.mean(cv_r2_scores):.4f}")
    
    # Fit the model on the full training data
    model.fit(X_train, y_train)
    
    # Predict on the test set
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Log parameters if available
    params = model.get_params() if hasattr(model, "get_params") else "Not available"
    
    return {
        "rmse": np.mean(cv_rmse_scores),
        "mae": np.mean(cv_mae_scores),
        "r2": np.mean(cv_r2_scores),
        "params": params
    }, model

def train_custom_nn(X_train, X_test, y_train, y_test, config):
    """
    Train a custom neural network with the specified configuration.

    Parameters:
        X_train (ndarray): Training features.
        X_test (ndarray): Testing features (used as validation data).
        y_train (ndarray): Training labels.
        y_test (ndarray): Testing labels.
        config (dict): Configuration dictionary for the model.

    Returns:
        dict: Metrics and model.
    """
    # Get the model from the config (which is a KerasRegressor)
    model = config.get("model")

    # Ensure the model is instantiated correctly
    if model is None:
        raise ValueError("Model configuration is missing or incorrect.")
    
    # Get the configuration values for training parameters
    epochs = model.epochs
    batch_size = model.batch_size

    # If the model is an instance of KerasRegressor, we need to train it with .fit() method
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Fit the model with validation data from X_test and y_test
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
              batch_size=batch_size, callbacks=[early_stopping], verbose=1, shuffle=False)
    
    history = model.history_
    
    # Make predictions on the test set
    predictions = model.predict(X_test)
    
    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Return metrics and the trained model
    return {"rmse": rmse, "mae": mae, "r2": r2, "params": {
        "epochs": epochs,
        "batch_size": batch_size
    }}, model, history


# not needed and not working properly
'''def train_custom_nn(X_train, X_test, y_train, y_test, config, cv_splits=5):
    """
    Train a custom neural network with the specified configuration.

    Parameters:
        X_train (ndarray): Training features.
        X_test (ndarray): Testing features.
        y_train (ndarray): Training labels.
        y_test (ndarray): Testing labels.
        config (dict): Configuration dictionary for the model.

    Returns:
        dict: Metrics and model.
    """
    # Get the model from the config (which is a KerasRegressor)
    model = config.get("model")

    # Ensure the model is instantiated correctly
    if model is None:
        raise ValueError("Model configuration is missing or incorrect.")
    
    # Get the configuration values for training parameters
    epochs = model.epochs
    batch_size = model.batch_size

    # Set up TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # Store metrics for each fold
    fold_metrics = []

    # Perform TimeSeriesCrossValidation
    for train_index, val_index in tscv.split(X_train):
        # Split data into training and validation sets for the current fold
        X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

        # If the model is an instance of KerasRegressor, we need to train it with .fit() method
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Fit the model
        model.fit(X_train_cv, y_train_cv, validation_data=(X_val_cv, y_val_cv), epochs=epochs,
                  batch_size=batch_size, callbacks=[early_stopping], verbose=0)

        # Make predictions on the validation set
        predictions = model.predict(X_val_cv)

        # Calculate evaluation metrics for this fold
        rmse = np.sqrt(mean_squared_error(y_val_cv, predictions))
        mae = mean_absolute_error(y_val_cv, predictions)
        r2 = r2_score(y_val_cv, predictions)

        # Store the metrics for this fold
        fold_metrics.append({"rmse": rmse, "mae": mae, "r2": r2})

    # Calculate average metrics across all folds
    avg_rmse = np.mean([metrics["rmse"] for metrics in fold_metrics])
    avg_mae = np.mean([metrics["mae"] for metrics in fold_metrics])
    avg_r2 = np.mean([metrics["r2"] for metrics in fold_metrics])

    print(f"Metrics for {config.get('name', 'Custom NN')}:")
    print(f"Average RMSE: {avg_rmse:.4f}, Average MAE: {avg_mae:.4f}, Average R²: {avg_r2:.4f}")

    # Return metrics and the trained model
    return {"rmse": avg_rmse, "mae": avg_mae, "r2": avg_r2, "params": {
        "epochs": epochs,
        "batch_size": batch_size
    }}, model'''

# Main pipeline function
def main_pipeline(dataset_path, target_column, models_config, sample_size=None):
    """
    Main pipeline for training and evaluating multiple models on a dataset.

    Parameters:
        dataset_path (str): Path to the dataset CSV file.
        target_column (str): The target variable for prediction.
        models_config (dict): Configuration dictionary for models.
        sample_size (int, optional): Number of samples to use from the dataset.

    Returns:
        results_df (pd.DataFrame): DataFrame containing model performance metrics.
        best_model: The best-performing model.
        history: Dictionary with all models' training details and metrics.
    """
    # Load dataset and prepare data
    df = pd.read_csv(dataset_path)
    X_train, X_test, y_train, y_test = prepare_data(df, target_column, sample_size=sample_size)

    results = []
    trained_models = {}
    model_history = {}

    for name, config in models_config.items():
        print(f"\nTraining model: {name}")
        model = config.get("model")
        optimized = config.get("grid_search", False)
        param_grid = config.get("param_grid", None)

        try:
            if name.startswith("Custom NN"):
                # Train a custom neural network
                metrics, trained_model, history = train_custom_nn(
                    X_train.values, X_test.values, y_train, y_test.values, config
                )
                epochs = metrics.get("params", {}).get("epochs", np.nan)
                best_params = metrics.get("params", None)
                tested_models = "Not applicable"
            
            elif name in ["XGBoost", "Random Forest", "MLP Regressor"]:
                # Train traditional ML models
                metrics, trained_model = train_model(
                    model, X_train, y_train, X_test, y_test, optimized, param_grid
                )
                if name == "XGBoost":
                    epochs = trained_model.get_params().get("n_estimators", np.nan)
                elif name == "Random Forest":
                    epochs = trained_model.get_params().get("n_estimators", np.nan)
                elif name == "MLP Regressor":
                    epochs = trained_model.get_params().get("max_iter", np.nan)
                else:
                    epochs = np.nan
                best_params = metrics.get("params", None)
                tested_models = (
                    trained_model.cv_results_ if isinstance(trained_model, GridSearchCV) else "Not applicable"
                )
            
            else:
                # Other models
                metrics, trained_model = train_model(
                    model, X_train, y_train, X_test, y_test, optimized, param_grid
                )
                epochs = np.nan
                best_params = metrics.get("params", None)
                tested_models = "Not applicable"

            # Append results
            results.append(
                {
                    "model_name": name,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "epochs": epochs,
                    "best_params": best_params,
                    "history": pd.DataFrame(history),
                    "tested_models": tested_models,
                }
            )
            trained_models[name] = trained_model
            model_history[name] = metrics

            print(
                f"{name} Metrics: RMSE={metrics['rmse']}, MAE={metrics['mae']}, R2={metrics['r2']}, Epochs={epochs}"
            )

        except Exception as e:
            print(f"Error training model {name}: {e}")
            continue

    # Convert results to a DataFrame and sort by RMSE
    results_df = pd.DataFrame(results).sort_values(by="rmse")
    best_model_name = results_df.iloc[0]["model_name"]
    print(f"\nBest model: {best_model_name} with RMSE = {results_df.iloc[0]['rmse']}")

    return results_df, trained_models.get(best_model_name, None), model_history