# Digital_HealthTwin

Digital_HealthTwin is a modular AI-based clinical decision support prototype that integrates document intelligence, multivariate time-series forecasting, and structured recommendation generation into a unified RESTful service.

The system is designed as an end-to-end applied machine learning pipeline that demonstrates:

* Prescription image processing using OCR
* Structured medical data extraction
* Multivariate health risk forecasting using LSTM models
* Deviation-based risk quantification
* LLM-assisted preventive recommendation generation
* Modular API-based system integration

---

## System Architecture

Digital_HealthTwin follows a modular architecture:

1. **Document Intelligence Module**
   Prescription image → OCR → Structured entity extraction (60+ medical parameters)

2. **Time-Series Forecasting Module**
   Multivariate longitudinal health data (32 parameters) → 2-layer LSTM → Future value prediction

3. **Risk Quantification Engine**
   Predicted vitals → Deviation from medically defined safe ranges → Normalized risk score

4. **Recommendation Engine**
   Abnormal vitals + patient metadata → Structured preventive recommendations via LLM integration

5. **Unified REST API Layer**
   All components exposed through a Flask-based microservice

---

## Key Features

* Extraction of 60+ structured medical entities from prescription documents
* 32-parameter multivariate time-series modeling
* Synthetic longitudinal dataset generation (50 users × 52 weeks)
* 2-layer LSTM architecture for forecasting
* Per-parameter scaling and inverse transformation
* Deviation-based risk scoring logic
* LLM-driven structured preventive guidance
* Modular and endpoint-based API design

---

## API Endpoints

The unified application exposes the following REST endpoints:

| Endpoint                | Method | Description                                                          | Input                                        | Output                                                        |
| ----------------------- | ------ | -------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------- |
| `/`                     | GET    | Base route to verify service is running                              | None                                         | Welcome message                                               |
| `/process-prescription` | POST   | Performs OCR and structured data extraction from prescription images | Image file (multipart/form-data)             | Extracted doctor info, vitals, complaints, medications (JSON) |
| `/calculate-risk`       | POST   | Forecasts vitals and computes health risk score                      | `input_sequence` (5 × 32 multivariate array) | `risk_score` (numeric)                                        |
| `/get-recommendations`  | POST   | Generates structured preventive recommendations                      | `vitals` + `body_params` (JSON)              | Structured recommendation response                            |
| `/health`               | GET    | Health-check endpoint                                                | None                                         | Service status                                                |

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/SoudeepGhoshal/Digital_HealthTwin.git
cd Digital_HealthTwin
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## System Requirements

### Python

* Python 3.9+

### Tesseract OCR

**Linux (Ubuntu):**

```bash
sudo apt install tesseract-ocr
```

**Windows:**

* Install from: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
* Default path expected: `C:\Program Files\Tesseract-OCR\tesseract.exe`

---

## Running the Application

Start the Flask server:

```bash
python app.py
```

Server will run at:

```
http://localhost:5002
```

---

## Example Requests

### OCR Prescription Processing

```bash
POST /process-prescription
Content-Type: multipart/form-data
```

### Risk Prediction

```json
POST /calculate-risk
{
  "input_sequence": [[...32 parameters...], ... 5 time steps]
}
```

### Recommendation Generation

```json
POST /get-recommendations
{
  "vitals": { ... },
  "body_params": { ... }
}
```

---

## Project Structure

```
Digital_HealthTwin/
│
├── app.py
├── calc_risk.py
├── ocr.py
├── gen.py
├── preprocess_train.py
├── gen_data.py
├── test.py
├── requirements.txt
│
└── res/
    ├── rnn_model_multivariate.keras
    ├── scalers.joblib
    ├── synthetic_health_data.csv
    ├── digital1.png
    └── image3.png
```

---

## Design Philosophy

Digital_HealthTwin is structured as a demonstrative applied AI system that emphasizes:

* Modular design
* Clean separation of responsibilities
* Structured JSON communication between components
* Reproducible ML pipelines
* Realistic health-data simulation
* Scalable API-based integration

The system is intended as a prototype showcasing end-to-end integration of computer vision, time-series modeling, and AI-assisted decision support.

---

## Limitations

* Health data is synthetically generated for modeling purposes
* Risk scoring is deviation-based and not clinically validated
* LLM recommendations are guidance-oriented and not medical advice

---

## Author

Soudeep Ghoshal
