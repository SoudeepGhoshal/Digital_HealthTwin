import numpy as np
from tensorflow.keras.models import load_model
from joblib import load

MODEL_PATH = 'res/rnn_model_multivariate.keras'
SCALER_PATH = 'res/scalers.joblib'

PARAMETERS = {
    "Blood Pressure": (90, 140),  # Systolic BP in mmHg
    "PT/INR": (0.8, 1.2),
    "B-type Natriuretic Peptide": (0, 100),  # pg/mL
    "Sodium": (135, 145),  # mmol/L
    "Potassium": (3.5, 5.0),  # mmol/L
    "Blood Glucose": (70, 140),  # mg/dL
    "HbA1c": (4.0, 5.6),  # %
    "Lipid Profile": (150, 240),  # Total cholesterol in mg/dL
    "Renal Function Tests": (0.6, 1.2),  # Creatinine in mg/dL
    "Creatinine": (0.6, 1.2),  # mg/dL
    "eGFR": (60, 120),  # mL/min/1.73m²
    "Urinalysis": (0, 1),  # Binary for abnormalities
    "Liver Function Tests": (10, 40),  # ALT in U/L
    "viral_markers": (0, 1),  # Binary for infections
    "tsh": (0.4, 4.0),  # µIU/mL
    "free_t3": (2.3, 4.1),  # pg/mL
    "free_t4": (0.8, 2.7),  # ng/dL
    "crp": (0, 10),  # mg/L
    "CBC": (4.5, 11),  # White blood cell count in x10^3/µL
    "Ferritin": (12, 300),  # ng/mL
    "Iron Studies": (50, 175),  # Serum iron in µg/dL
    "ECG": (0, 1),  # Binary for abnormalities
    "Cardiac Enzymes": (0, 40),  # Troponin in ng/L
    "Bone Density Test": (-2.5, 1.0),  # T-score
    "Calcium Levels": (8.5, 10.2),  # mg/dL
    "Vitamin D": (20, 50),  # ng/mL
    "visual_acuity": (20, 20),  # Perfect vision (dummy value)
    "slit_lamp": (0, 1),  # Binary for abnormalities
    "retinal_exam": (0, 1),  # Binary for abnormalities
    "tonometry": (10, 21),  # mmHg
    "Culture Tests": (0, 1),  # Binary for results
    "Contrast Sensitivity Test": (1.5, 2.5),  # LogMAR
}

def predict_value(input_sequence, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    # Load the saved MinMaxScaler(s) for all parameters
    scalers = load(scaler_path)  # Dictionary of scalers, one per parameter

    # Convert input_sequence to a numpy array
    input_sequence = np.array(input_sequence)  # Shape: (sequence_length, num_parameters)
    if input_sequence.shape != (5, 32):
        raise ValueError(f"Input sequence must have shape (5, 32), but got {input_sequence.shape}")

    # Apply MinMax scaling to the input sequence (scale each parameter individually)
    parameter_names = [
        'Blood Pressure', 'PT/INR', 'B-type Natriuretic Peptide', 'Sodium', 'Potassium',
        'Blood Glucose', 'HbA1c', 'Lipid Profile', 'Renal Function Tests', 'Creatinine',
        'eGFR', 'Urinalysis', 'Liver Function Tests', 'viral_markers', 'tsh', 'free_t3',
        'free_t4', 'crp', 'CBC', 'Ferritin', 'Iron Studies', 'ECG', 'Cardiac Enzymes',
        'Bone Density Test', 'Calcium Levels', 'Vitamin D', 'visual_acuity', 'slit_lamp',
        'retinal_exam', 'tonometry', 'Culture Tests', 'Contrast Sensitivity Test'
    ]

    scaled_sequence = np.zeros_like(input_sequence)
    for i, param_name in enumerate(parameter_names):  # Loop through each parameter
        if param_name not in scalers:  # Handle missing keys
            raise KeyError(f"Scaler for parameter '{param_name}' not found in scalers dictionary.")
        scaled_sequence[:, i] = scalers[param_name].transform(input_sequence[:, i].reshape(-1, 1)).flatten()

    # Reshape scaled_sequence to match the model input shape: (1, sequence_length, num_parameters)
    #scaled_sequence = np.expand_dims(scaled_sequence, axis=0)
    #scaled_sequence = scaled_sequence.reshape(1, 5, 32)

    # Load the trained LSTM model
    model = load_model(model_path)

    # Predict the next values using the trained model
    scaled_predictions = model.predict(scaled_sequence, verbose=0)  # Shape: (1, num_parameters)
    scaled_predictions = scaled_predictions.flatten()

    # Reverse MinMax scaling for the predicted values
    predictions = []
    for i, param in enumerate(scalers.keys()):  # Iterate over each parameter
        predictions.append(scalers[param].inverse_transform([[scaled_predictions[i]]])[0, 0])

    predictions = [float(value) for value in predictions]
    return predictions

def calculate_risk(input, safe_ranges=PARAMETERS):
    predicted_values = predict_value(input)
    parameter_risks = {}
    total_risk = 0

    for i, (param, limits) in enumerate(safe_ranges.items()):
        lower_limit, upper_limit = limits
        mean_safe_value = (lower_limit + upper_limit) / 2  # Calculate mean safe value
        half_range = (upper_limit - lower_limit) / 2  # Half of the safe range

        # Calculate deviation from the mean safe value
        deviation = abs(predicted_values[i] - mean_safe_value)

        # Normalize deviation as percentage of risk
        if half_range > 0:
            risk = (deviation / half_range) * 100
        else:
            risk = 0  # No risk if the range is invalid

        # Cap the risk at 100%
        risk = min(risk, 100)

        # Add to parameter-specific and total risks
        parameter_risks[param] = risk
        total_risk += risk

    # Calculate overall risk as the average risk across all parameters
    overall_risk = round(total_risk / len(safe_ranges), 2)

    return overall_risk

sample_input = [
    [120, 1.0, 150, 140, 4.0, 100, 5.5, 200, 1.2, 0.9, 90, 4.5, 70, 1.1, 2.0, 3.0, 1.8, 0.6, 12, 50, 30, 70, 1.0, 1.2, 9.5, 25, 20, 3.2, 2.5, 15, 1.0, 3.1],  # Time step 1
    [125, 1.1, 160, 138, 4.2, 105, 5.6, 210, 1.3, 0.8, 85, 4.6, 75, 1.2, 2.1, 3.1, 1.7, 0.7, 13, 52, 31, 72, 1.1, 1.3, 9.6, 26, 21, 3.3, 2.4, 16, 1.1, 3.2],  # Time step 2
    [122, 1.2, 155, 137, 4.1, 102, 5.4, 205, 1.1, 0.7, 88, 4.4, 72, 1.0, 2.2, 3.2, 1.9, 0.5, 11, 51, 29, 73, 1.0, 1.1, 9.4, 24, 22, 3.4, 2.6, 14, 1.2, 3.3],  # Time step 3
    [123, 1.3, 158, 139, 4.3, 103, 5.7, 208, 1.4, 0.6, 87, 4.3, 73, 1.3, 2.3, 3.0, 1.6, 0.8, 14, 53, 30, 74, 1.2, 1.4, 9.7, 27, 23, 3.5, 2.7, 17, 1.3, 3.4],  # Time step 4
    [124, 1.4, 157, 136, 4.4, 104, 5.8, 207, 1.5, 0.5, 86, 4.2, 74, 1.4, 2.4, 3.3, 1.5, 0.9, 15, 54, 31, 75, 1.3, 1.5, 9.8, 28, 24, 3.6, 2.8, 18, 1.4, 3.5]  # Time step 5
]

sample_input_2 = [
    [250, 8.0, 1000, 180, 9.0, 500, 15.0, 500, 10.0, 15.0, 200, 50.0, 200, 5.0, 20.0, 20.0, 15.0, 50.0, 100, 500, 200, 5.0, 100, 10.0, 50, 100, 50, 10.0, 10.0, 100, 5.0, 50.0],  # Timestamp 1
    [240, 7.5, 950, 175, 8.5, 480, 14.5, 480, 9.5, 14.5, 190, 48.0, 190, 4.5, 19.0, 19.0, 14.0, 48.0, 95, 480, 190, 4.5, 95, 9.5, 48, 95, 48, 9.5, 9.5, 95, 4.5, 48.0],  # Timestamp 2
    [230, 7.0, 900, 170, 8.0, 460, 14.0, 460, 9.0, 14.0, 180, 46.0, 180, 4.0, 18.0, 18.0, 13.5, 46.0, 90, 460, 180, 4.0, 90, 9.0, 46, 90, 46, 9.0, 9.0, 90, 4.0, 46.0],  # Timestamp 3
    [220, 6.5, 850, 165, 7.5, 440, 13.5, 440, 8.5, 13.5, 170, 44.0, 170, 3.5, 17.0, 17.0, 13.0, 44.0, 85, 440, 170, 3.5, 85, 8.5, 44, 85, 44, 8.5, 8.5, 85, 3.5, 44.0],  # Timestamp 4
    [210, 6.0, 800, 160, 7.0, 420, 13.0, 420, 8.0, 13.0, 160, 42.0, 160, 3.0, 16.0, 16.0, 12.5, 42.0, 80, 420, 160, 3.0, 80, 8.0, 42, 80, 42, 8.0, 8.0, 80, 3.0, 42.0],  # Timestamp 5
]

print(calculate_risk(sample_input))