import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY environment variable not set")

from flask import jsonify


def analyze_health_status(vitals, body_params):
    """
    Analyze health status using available vitals and body parameters via Mistral API.
    Returns technical preventive measures based on the provided parameters.
    """
    # [Previous code for normal ranges and concerns remains the same until the prompt]
    normal_ranges = {
        "spo2": (95, 100),
        "heart_rate": (60, 100),
        "systolic": (90, 120),
        "diastolic": (60, 80),
        "temperature": (36.1, 37.2),
        "respiratory_rate": (12, 20),
        "bmi": (18.5, 24.9)
    }

    # Initialize concerns list
    concerns = []

    # Check each vital against normal ranges if available
    if 'spo2' in vitals and isinstance(vitals['spo2'], (int, float)):
        if vitals['spo2'] < normal_ranges['spo2'][0]:
            concerns.append(f"SpO2 below {normal_ranges['spo2'][0]}% (Current: {vitals['spo2']}%)")

    if 'heart_rate' in vitals and isinstance(vitals['heart_rate'], (int, float)):
        if not (normal_ranges['heart_rate'][0] <= vitals['heart_rate'] <= normal_ranges['heart_rate'][1]):
            concerns.append(f"HR outside range {normal_ranges['heart_rate'][0]}-{normal_ranges['heart_rate'][1]} bpm (Current: {vitals['heart_rate']} bpm)")

    if 'blood_pressure' in vitals and isinstance(vitals['blood_pressure'], (list, tuple)):
        if len(vitals['blood_pressure']) == 2:
            if vitals['blood_pressure'][0] > normal_ranges['systolic'][1]:
                concerns.append(f"Systolic BP elevated > {normal_ranges['systolic'][1]} mmHg (Current: {vitals['blood_pressure'][0]} mmHg)")
            if vitals['blood_pressure'][1] > normal_ranges['diastolic'][1]:
                concerns.append(f"Diastolic BP elevated > {normal_ranges['diastolic'][1]} mmHg (Current: {vitals['blood_pressure'][1]} mmHg)")

    if 'temperature' in vitals and isinstance(vitals['temperature'], (int, float)):
        if not (normal_ranges['temperature'][0] <= vitals['temperature'] <= normal_ranges['temperature'][1]):
            concerns.append(f"Temperature deviation from {normal_ranges['temperature'][0]}-{normal_ranges['temperature'][1]}°C (Current: {vitals['temperature']}°C)")

    if 'respiratory_rate' in vitals and isinstance(vitals['respiratory_rate'], (int, float)):
        if not (normal_ranges['respiratory_rate'][0] <= vitals['respiratory_rate'] <= normal_ranges['respiratory_rate'][1]):
            concerns.append(f"RR outside range {normal_ranges['respiratory_rate'][0]}-{normal_ranges['respiratory_rate'][1]} breaths/min (Current: {vitals['respiratory_rate']})")

    try:
        MISTRAL_API_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY environment variable not set")

        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

        # Create vital signs section
        vital_signs_list = []
        for vital, value in vitals.items():
            if vital == 'blood_pressure' and isinstance(value, (tuple, list)) and len(value) == 2:
                vital_signs_list.append(f"- Blood Pressure: {value[0]}/{value[1]} mmHg")
            else:
                vital_name = ' '.join(word.capitalize() for word in vital.split('_'))
                vital_signs_list.append(f"- {vital_name}: {value}")

        # Create body parameters section
        body_params_list = []
        for param, value in body_params.items():
            param_name = ' '.join(word.capitalize() for word in param.split('_'))
            body_params_list.append(f"- {param_name}: {value}")

        # Structured prompt for categorical recommendations
        prompt = f"""
        As a clinical decision support system, analyze the following patient data and provide structured preventive measures.
        Format your response exactly as shown below, with 4-5 main categories and specific measures under each category.
        Each category should be numbered and in Title Case, followed by specific recommendations on new lines starting with bullets (-).

        Patient Data:

        Vital Signs:
        {chr(10).join(vital_signs_list)}

        Patient Metrics:
        {chr(10).join(body_params_list)}

        Required Response Format Example:
        1. Monitor Vital Signs:
        - Set up daily blood pressure monitoring schedule
        - Track heart rate variability patterns

        2. Medication Management:
        - Review current prescription interactions
        - Establish optimal dosing schedule

        Provide your recommendations following this exact format, focusing on:
        - Technical medical interventions
        - Specific monitoring protocols
        - Clear action items
        - Quantifiable measures where possible
        - Evidence-based preventive strategies
        
        Make exactly three broad title:
        - Monitor Vital Signs
        - Medication Management
        - Lifestyle Modifications

        Make sure each category has 1-2 specific bullet points with detailed technical recommendations in around 20 words.
        """

        payload = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 50000
        }

        response = requests.post(MISTRAL_API_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        assessment = result['choices'][0]['message']['content']

        # Parse the response into a dictionary
        preventive_measures = {}
        current_category = None
        current_measures = []

        for line in assessment.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Check for numbered category (e.g., "1. Category Name:")
            if line[0].isdigit() and '. ' in line:
                # Save previous category if it exists
                if current_category and current_measures:
                    preventive_measures[current_category] = current_measures

                # Start new category
                current_category = line.split('. ', 1)[1].rstrip(':')
                current_measures = []

            # Check for bullet points
            elif line.startswith('- '):
                if current_category:
                    current_measures.append(line[2:])

        # Add the last category
        if current_category and current_measures:
            preventive_measures[current_category] = current_measures

        return preventive_measures

    except requests.exceptions.RequestException as e:
        return {"Error": f"Error connecting to Mistral API: {str(e)}"}
    except Exception as e:
        return {"Error": f"An error occurred: {str(e)}"}

def get_recom(vitals, params):
    measures = analyze_health_status(vitals, params)
    return jsonify(measures)

# Example usage
if __name__ == "__main__":
    os.environ["MISTRAL_API_KEY"] = "zV19HjmfkSCA4hv3cXR27rYLjW3Bw3pU"
    # Example with some concerning vital signs
    sample_vitals = {
        "heart_rate": 110,  # Elevated
        "blood_pressure": (150, 95),  # Hypertensive
        "spo2": 93,  # Below normal
        "respiratory_rate": 22  # Elevated
    }

    sample_body_params = {
        "age": 55,
        "gender": "M",
        "bmi": 31.2,  # Obese
        "weight": 95,
        "height": 175
    }