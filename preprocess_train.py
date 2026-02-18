import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from joblib import dump, load
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

FILE_PATH = 'res/synthetic_health_data.csv'
SEQUENCE_LENGTH = 5
MODEL_SAVE_PATH = 'res/rnn_model_multivariate.keras'
SCALERS_SAVE_PATH = 'res/scalers.joblib'

def preprocess(file_path=FILE_PATH, sequence_length=SEQUENCE_LENGTH):
    """
    Preprocesses the dataset for training an RNN model.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
        sequence_length (int): Length of the input sequences.

    Returns:
        X_train (numpy.ndarray): Training sequences.
        y_train (numpy.ndarray): Training targets.
        X_val (numpy.ndarray): Validation sequences.
        y_val (numpy.ndarray): Validation targets.
        scalers (dict): Dictionary of MinMaxScaler objects for each parameter.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Initialize a dictionary to store scaled data for each parameter
    scaled_data = {}
    scalers = {}

    # Get unique parameters
    parameters = data["Parameter"].unique()

    # Apply Min-Max Scaling for each parameter
    for param in parameters:
        # Filter data for the current parameter
        param_data = data[data["Parameter"] == param].copy()  # Use .copy() to avoid modifying the original DataFrame

        # Apply Min-Max Scaling
        scaler = MinMaxScaler()
        param_data["Scaled Value"] = scaler.fit_transform(param_data[["Value"]])

        # Store the scaled data and scaler
        scaled_data[param] = param_data
        scalers[param] = scaler

    # Combine scaled data into a single DataFrame
    scaled_data_df = pd.concat(scaled_data.values(), ignore_index=True)

    # Sort by User ID, Parameter, and Week to maintain sequence order
    scaled_data_df = scaled_data_df.sort_values(by=["User ID", "Parameter", "Week"])

    # Function to prepare sequences and targets
    def create_sequences(data, sequence_length=5):
        sequences = []
        targets = []

        for user_id in data["User ID"].unique():
            for param in data["Parameter"].unique():
                # Filter data for the specific user and parameter
                user_param_data = data[(data["User ID"] == user_id) & (data["Parameter"] == param)]
                scaled_values = user_param_data["Scaled Value"].values

                # Create sequences and targets
                for i in range(len(scaled_values) - sequence_length):
                    sequences.append(scaled_values[i:i + sequence_length])
                    targets.append(scaled_values[i + sequence_length])

        return np.array(sequences), np.array(targets)

    # Prepare sequences and targets
    X, y = create_sequences(scaled_data_df, sequence_length=sequence_length)

    # Train-test split (80% training, 20% testing)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure X_train and X_val are 3D
    if len(X_train.shape) == 2:
        X_train = np.expand_dims(X_train, axis=-1)
    if len(X_val.shape) == 2:
        X_val = np.expand_dims(X_val, axis=-1)

    return X_train, y_train, X_val, y_val, scalers

def train_model(X_train, y_train, X_val, y_val, num_parameters=32, epochs=50, batch_size=32, model_save_path=MODEL_SAVE_PATH):
    """
    Trains an LSTM-based RNN model for multivariate time-series prediction.

    Args:
        X_train (numpy.ndarray): Training input sequences (shape: num_samples, sequence_length, num_parameters).
        y_train (numpy.ndarray): Training targets (shape: num_samples, num_parameters).
        X_val (numpy.ndarray): Validation input sequences (same shape as X_train).
        y_val (numpy.ndarray): Validation targets (same shape as y_train).
        num_parameters (int): Number of parameters to predict.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        model_save_path (str): File path to save the trained model.

    Returns:
        model (Sequential): Trained LSTM model.
    """
    # Define input shape for the model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_parameters)

    # Define the LSTM-based RNN model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    # Dense layer with 'num_parameters' units to predict all parameters at the next time step
    model.add(Dense(num_parameters))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # Model Checkpoint to save the best model during training
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint],
        verbose=1
    )

    print(f"Model training completed. Best model saved to {model_save_path}.")

    return model

def eval_model():
    best_model = load_model(MODEL_SAVE_PATH)

    # Evaluate on the test dataset
    test_loss, test_mae = best_model.evaluate(X_val, y_val, verbose=1)

    print(f"Test Loss: {test_loss}")
    print(f"Test Mean Absolute Error: {test_mae}")

def evaluate_model(X_val, y_val, model_path=MODEL_SAVE_PATH, scaler_path=SCALERS_SAVE_PATH):
    """
    Loads a trained model and evaluates it on the validation set to return accuracy.

    Args:
        X_val (numpy array): Validation data (features).
        y_val (numpy array): Validation data (targets).
        model_path (str): Path to the saved LSTM model.
        scaler_path (str): Path to the saved scalers (joblib file).

    Returns:
        float: Accuracy in percentage.
    """
    # Load the trained LSTM model
    model = load_model(model_path)

    # Load the scalers (if needed for scaling predictions back to original scale)
    scalers = load(scaler_path)

    # Make predictions
    predictions = model.predict(X_val, verbose=0)

    if y_val.ndim == 1:
        y_val = y_val.reshape(predictions.shape)

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_val - predictions) / y_val)) * 100  # MAPE in percentage
    accuracy = 100 - mape  # Accuracy in percentage

    print(f"Validation MAPE: {mape:.2f}%")
    print(f"Validation Accuracy: {accuracy:.2f}%")

    return accuracy

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, scalers = preprocess()
    dump(scalers, SCALERS_SAVE_PATH)
    print(f"Scalers saved successfully")
    model = train_model(X_train, y_train, X_val, y_val)
    #ABG = evaluate_model(X_val, y_val)