from flask import Flask, request, jsonify
import os
import numpy as np
from werkzeug.utils import secure_filename

from ocr import get_ocr, extract_details
from calc_risk import calculate_risk
import gen_recom


app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Default
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Digital HealthTwin System!'})


# Prescription OCR and Parsing
@app.route('/process-prescription', methods=['POST'])
def process_prescription():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Check if a filename was submitted
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check if the file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image using imported functions from ocr.py
        ocr_text = get_ocr(filepath)
        extracted_data = extract_details(ocr_text)

        # Clean up - remove the uploaded file
        os.remove(filepath)

        # Return the results
        return jsonify(extracted_data)

    except Exception as e:
        # Clean up in case of error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500


# LSTM Health Risk Predict
@app.route('/calculate-risk', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()

        # Validate input data
        if not data or 'input_sequence' not in data:
            return jsonify({'error': 'No input_sequence provided'}), 400

        input_sequence = data['input_sequence']

        # Validate input sequence shape and type
        if not isinstance(input_sequence, list) or len(input_sequence) != 5:
            return jsonify({'error': 'Input sequence must be a list of 5 time steps'}), 400

        for step in input_sequence:
            if not isinstance(step, list) or len(step) != 32:
                return jsonify({'error': 'Each time step must contain 32 parameters'}), 400

        # Convert input to correct format
        input_array = np.array(input_sequence)

        # Calculate risk
        risk_score = calculate_risk(input_array)

        return jsonify({
            'risk_score': risk_score
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Health Recommendation
@app.route('/get-recommendations', methods=['POST'])
def get_recommendations():
    try:
        # Parse the incoming JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid input: No JSON data provided"}), 400

        # Extract vitals and body parameters
        vitals = data.get("vitals")
        body_params = data.get("body_params")
        if not vitals or not body_params:
            return jsonify({"error": "Invalid input: Missing 'vitals' or 'body_params'"}), 400

        # Call the function from gen.py
        response = gen_recom.get_recom(vitals, body_params)
        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
