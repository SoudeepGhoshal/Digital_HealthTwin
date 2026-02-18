import requests
import json
import sys

url = 'http://localhost:5002'


# OCR System Check
try:
    with open('res/digital1.png', 'rb') as img:
        files = {'file': img}
        response = requests.post(f'{url}/process-prescription', files=files)

    response.raise_for_status()

    print("OCR Test successful!")
    print(json.dumps(response.json(), indent=4))

except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the server. Make sure Flask is running.")
    sys.exit(1)

except requests.exceptions.HTTPError as e:
    print(f"OCR HTTP Error: {e}")
    print(f"Response: {response.text}")
    sys.exit(1)

except ValueError:
    print("OCR Error: Response is not valid JSON.")
    print(response.text)
    sys.exit(1)

except FileNotFoundError:
    print("OCR Error: Image file not found.")
    sys.exit(1)

except Exception as e:
    print(f"OCR Unexpected Error: {str(e)}")
    sys.exit(1)


# Prediction System Check
sample_input = [
    [120, 1.0, 150, 140, 4.0, 100, 5.5, 200, 1.2, 0.9, 90, 4.5, 70, 1.1, 2.0, 3.0, 1.8, 0.6, 12, 50, 30, 70, 1.0,
     1.2, 9.5, 25, 20, 3.2, 2.5, 15, 1.0, 3.1],
    [125, 1.1, 160, 138, 4.2, 105, 5.6, 210, 1.3, 0.8, 85, 4.6, 75, 1.2, 2.1, 3.1, 1.7, 0.7, 13, 52, 31, 72, 1.1,
     1.3, 9.6, 26, 21, 3.3, 2.4, 16, 1.1, 3.2],
    [122, 1.2, 155, 137, 4.1, 102, 5.4, 205, 1.1, 0.7, 88, 4.4, 72, 1.0, 2.2, 3.2, 1.9, 0.5, 11, 51, 29, 73, 1.0,
     1.1, 9.4, 24, 22, 3.4, 2.6, 14, 1.2, 3.3],
    [123, 1.3, 158, 139, 4.3, 103, 5.7, 208, 1.4, 0.6, 87, 4.3, 73, 1.3, 2.3, 3.0, 1.6, 0.8, 14, 53, 30, 74, 1.2,
     1.4, 9.7, 27, 23, 3.5, 2.7, 17, 1.3, 3.4],
    [124, 1.4, 157, 136, 4.4, 104, 5.8, 207, 1.5, 0.5, 86, 4.2, 74, 1.4, 2.4, 3.3, 1.5, 0.9, 15, 54, 31, 75, 1.3,
     1.5, 9.8, 28, 24, 3.6, 2.8, 18, 1.4, 3.5]
]

payload = {'input_sequence': sample_input}

try:
    response = requests.post(f'{url}/calculate-risk', json=payload)
    response.raise_for_status()
    result = response.json()

    print("Prediction Test successful!")
    print(f"Risk score: {result['risk_score']}")

except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the server. Make sure the Flask server is running.")
    sys.exit(1)
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
    print(f"Response: {response.text}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1)


# Health Recommendation System Check
sample_data = {
    "vitals": {
        "heart_rate": 110,
        "blood_pressure": [150, 95],
        "spo2": 93,
        "respiratory_rate": 22
    },
    "body_params": {
        "age": 55,
        "gender": "M",
        "bmi": 31.2,
        "weight": 95,
        "height": 175
    }
}

try:
    response = requests.post(f'{url}/get-recommendations', json=sample_data)

    if response.status_code == 200:
        print("Recommendation Test successful!")
        print("Response from server:")
        print(json.dumps(response.json(), indent=4))
    else:
        print(f"Failed with status code {response.status_code}:")
        print(response.text)
except Exception as e:
    print(f"An error occurred: {str(e)}")