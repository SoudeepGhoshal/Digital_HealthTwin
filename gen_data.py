import numpy as np
import pandas as pd
import random

# Parameters and their reference ranges (example ranges)
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


# Function to generate a time-series trend for a parameter
def generate_time_series(base, variability, weeks=52, anomaly_chance=0.05):
    values = []
    for week in range(weeks):
        # Seasonal fluctuation (sine wave for periodic changes)
        seasonal = np.sin(2 * np.pi * week / 52) * variability * 0.5
        # Random noise
        noise = np.random.normal(0, variability * 0.1)
        # Base value with seasonal and random components
        value = base + seasonal + noise

        # Occasionally add anomalies
        if random.random() < anomaly_chance:
            value += np.random.normal(0, variability * 2)

        values.append(round(value, 2))
    return values


# Generate data for all users and parameters
def generate_dataset(num_users=50, weeks=52):
    data = []
    for user_id in range(1, num_users + 1):
        for param, (min_val, max_val) in PARAMETERS.items():
            # Each user has a personalized baseline for the parameter
            base = np.random.uniform(min_val, max_val)
            variability = (max_val - min_val) * 0.1  # 10% of the range
            time_series = generate_time_series(base, variability, weeks)

            # Add the data to the dataset
            for week, value in enumerate(time_series, start=1):
                data.append([user_id, week, param, value])

    # Convert to a DataFrame
    df = pd.DataFrame(data, columns=["User ID", "Week", "Parameter", "Value"])
    return df


# Generate the dataset
dataset = generate_dataset()

# Save to CSV
dataset.to_csv("res/synthetic_health_data.csv", index=False)
print("Dataset saved to 'synthetic_health_data.csv'.")
