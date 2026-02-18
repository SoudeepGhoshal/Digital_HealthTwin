# importing the libraries
import platform
import cv2
import pytesseract
import re
import json

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_PATH = 'res/digital1.png'

def get_ocr(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # For getting the text and number from image
    ocr_text = pytesseract.image_to_string(img)

    #print(ocr_text)

    return ocr_text

# Function to extract details
def extract_details(text):
    # Extract doctor's name
    doctor_name = re.match(r"(?:Dr\.?|DR\.?|Dr|DR)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})", text)
    doctor_name = doctor_name.group(1) if doctor_name else None

    # Extract doctor's designation
    possible_designations = [
        "MBBS", "MD", "M.D.", "MS", "DM", "MCh", "DNB", "BDS", "MDS", "PhD",
        "Consultant", "Surgeon", "General Physician", "Dermatologist",
        "Neurologist", "Cardiologist", "Oncologist", "Psychiatrist", "Endocrinologist",
        "Pediatrician", "Orthopedic Surgeon", "Urologist", "Gastroenterologist",
        "Nephrologist", "Rheumatologist", "Hematologist", "Pulmonologist",
        "Immunologist", "ENT Specialist", "Ophthalmologist", "Radiologist",
        "Anesthesiologist", "Pathologist", "Physiotherapist", "Chiropractor",
        "Occupational Therapist", "Speech Therapist", "Dietitian", "Nutritionist",
        "Psychologist", "Clinical Psychologist", "Gynecologist", "Obstetrician",
        "Hepatologist", "Sports Medicine Specialist", "Forensic Pathologist",
        "Infectious Disease Specialist", "Allergist", "Critical Care Specialist",
        "Emergency Medicine Specialist", "Neonatologist", "Family Medicine Physician",
        "Palliative Care Specialist", "Reproductive Endocrinologist",
        "Pain Management Specialist", "Hospitalist", "Vascular Surgeon",
        "Hand Surgeon", "Plastic Surgeon", "Cosmetic Surgeon", "Maxillofacial Surgeon",
        "Trauma Surgeon", "Bariatric Surgeon", "Podiatrist", "Epidemiologist",
        "Medical Oncologist", "Surgical Oncologist", "Radiation Oncologist",
        "Preventive Medicine Specialist", "Toxicologist", "Osteopath",
        "Geneticist", "Virologist", "Biotechnologist", "Microbiologist",
        "Pharmacologist", "Veterinarian", "Clinical Researcher",
        "Health Administrator", "Public Health Specialist", "Biostatistician",
        "Medical Informatician", "Biomedical Engineer", "Nurse Practitioner",
        "Midwife", "Clinical Nurse Specialist", "Nurse Anesthetist",
        "Cardiac Electrophysiologist", "Neuropsychiatrist", "Pediatric Surgeon",
        "Neurointerventional Radiologist", "Interventional Cardiologist",
        "Emergency Physician", "Critical Care Physician", "Ocular Oncologist"
    ]
    designation_pattern = r"\b(?:{})\b".format("|".join(re.escape(desig) for desig in possible_designations))
    designations = re.findall(designation_pattern, text)
    designations = list(set(designations))

    # Extract doctor's phone number
    doctor_phone = re.search(r"\s*(\d{10,13})", text)
    doctor_phone = doctor_phone.group(1).strip() if doctor_phone else None

    # Define vitals parameters and their units
    vitals_parameters = {
        "BP": {"regex": r"BP[:\s]+([\d/]+)", "unit": "mmHg"},
        "Pulse Rate": {"regex": r"Pulse Rate[:\s]+([\d]+)", "unit": "bpm"},
        "Pulse Pattern": {"regex": r"Pulse Pattern\s*:\s*([^\s]+)", "unit": ""},
        "Resp Rate": {"regex": r"Resp Rate[:\s]+([\d]+)", "unit": "cycles/min"},
        "Resp Pattern": {"regex": r"Resp Pattern\s*:\s*([^\s]+)", "unit": ""},
        "Temperature": {"regex": r"(?:Temperature)\s*[:\-]?\s*(\d{1,3})", "unit": "Celsius"},
        "SpO2": {"regex": r"SpO2[:\s]+([\d]+)", "unit": "%"},
        "RBS": {"regex": r"(?:RBS)\s*[:\-]?\s*(\d{1,3})", "unit": "%"},
        "Weight": {"regex": r"Weight[:\s]+([\d.]+)", "unit": "kg"},
        "BMI": {"regex": r"BMI[:\s]+([\d.]+)", "unit": ""},
        "Blood Pressure": {"regex": r"Blood Pressure[:\s]+([\d/]+)", "unit": "mmHg"},
        "PT/INR": {"regex": r"PT/INR[:\s]+([\d.]+)", "unit": ""},
        "B-type Natriuretic Peptide": {"regex": r"B-type Natriuretic Peptide[:\s]+([\d.]+)", "unit": "pg/mL"},
        "Sodium": {"regex": r"Sodium[:\s]+([\d.]+)", "unit": "mmol/L"},
        "Potassium": {"regex": r"Potassium[:\s]+([\d.]+)", "unit": "mmol/L"},
        "Blood Glucose": {"regex": r"Blood Glucose[:\s]+([\d.]+)", "unit": "mg/dL"},
        "HbA1c": {"regex": r"HbA1c[:\s]+([\d.]+)", "unit": "%"},
        "Creatinine": {"regex": r"Creatinine[:\s]+([\d.]+)", "unit": "mg/dL"},
        "eGFR": {"regex": r"eGFR[:\s]+([\d.]+)", "unit": "mL/min/1.73m²"},
        "Calcium Levels": {"regex": r"Calcium Levels[:\s]+([\d.]+)", "unit": "mg/dL"},
        "Vitamin D": {"regex": r"Vitamin D[:\s]+([\d.]+)", "unit": "ng/mL"},
        "Edema Level": {"regex": r"Edema Level[:\s]+([\d]+)", "unit": ""},
        "Triglycerides": {"regex": r"Triglycerides[:\s]+([\d.]+)", "unit": "mg/dL"},
        "HDL": {"regex": r"HDL[:\s]+([\d.]+)", "unit": "mg/dL"},
        "LDL": {"regex": r"LDL[:\s]+([\d.]+)", "unit": "mg/dL"},
        "VLDL": {"regex": r"VLDL[:\s]+([\d.]+)", "unit": "mg/dL"},
        "Total Cholesterol": {"regex": r"(Total Cholesterol|Cholesterol)[:\s]+([\d.]+)", "unit": "mg/dL"},
        "Albumin": {"regex": r"Albumin[:\s]+([\d.]+)", "unit": "g/dL"},
        "Globulin": {"regex": r"Globulin[:\s]+([\d.]+)", "unit": "g/dL"},
        "A/G Ratio": {"regex": r"A/G Ratio[:\s]+([\d.]+)", "unit": ""},
        "Bilirubin": {"regex": r"Bilirubin[:\s]+([\d.]+)", "unit": "mg/dL"},
        "Alkaline Phosphatase": {"regex": r"Alkaline Phosphatase[:\s]+([\d.]+)", "unit": "U/L"},
        "AST": {"regex": r"AST[:\s]+([\d.]+)", "unit": "U/L"},
        "ALT": {"regex": r"ALT[:\s]+([\d.]+)", "unit": "U/L"},
        "ESR": {"regex": r"ESR[:\s]+([\d.]+)", "unit": "mm/hr"},
        "WBC Count": {"regex": r"WBC Count[:\s]+([\d.]+)", "unit": "cells/μL"},
        "RBC Count": {"regex": r"RBC Count[:\s]+([\d.]+)", "unit": "million/μL"},
        "Platelet Count": {"regex": r"Platelet Count[:\s]+([\d.]+)", "unit": "cells/μL"},
        "Hemoglobin": {"regex": r"Hemoglobin[:\s]+([\d.]+)", "unit": "g/dL"},
        "MCV": {"regex": r"MCV[:\s]+([\d.]+)", "unit": "fL"},
        "MCH": {"regex": r"MCH[:\s]+([\d.]+)", "unit": "pg"},
        "MCHC": {"regex": r"MCHC[:\s]+([\d.]+)", "unit": "g/dL"},
        "RDW": {"regex": r"RDW[:\s]+([\d.]+)", "unit": "%"},
        "Serum Calcium": {"regex": r"Serum Calcium[:\s]+([\d.]+)", "unit": "mg/dL"},
        "Serum Magnesium": {"regex": r"Serum Magnesium[:\s]+([\d.]+)", "unit": "mg/dL"},
        "Serum Phosphate": {"regex": r"Serum Phosphate[:\s]+([\d.]+)", "unit": "mg/dL"},
        "Uric Acid": {"regex": r"Uric Acid[:\s]+([\d.]+)", "unit": "mg/dL"},
        "Folic Acid": {"regex": r"Folic Acid[:\s]+([\d.]+)", "unit": "ng/mL"},
        "Vitamin B12": {"regex": r"Vitamin B12[:\s]+([\d.]+)", "unit": "pg/mL"},
        "TSH 3rd Generation": {"regex": r"TSH 3rd Generation[:\s]+([\d.]+)", "unit": "mIU/L"},
        "Free Testosterone": {"regex": r"Free Testosterone[:\s]+([\d.]+)", "unit": "pg/mL"},
        "Prostate-Specific Antigen": {"regex": r"Prostate-Specific Antigen[:\s]+([\d.]+)", "unit": "ng/mL"},
        "D-Dimer": {"regex": r"D-Dimer[:\s]+([\d.]+)", "unit": "ng/mL"},
        "Cortisol": {"regex": r"Cortisol[:\s]+([\d.]+)", "unit": "μg/dL"},
        "Amylase": {"regex": r"Amylase[:\s]+([\d.]+)", "unit": "U/L"},
        "Lipase": {"regex": r"Lipase[:\s]+([\d.]+)", "unit": "U/L"},
        "Serum Iron": {"regex": r"Serum Iron[:\s]+([\d.]+)", "unit": "μg/dL"},
        "Total Iron-Binding Capacity": {"regex": r"Total Iron-Binding Capacity[:\s]+([\d.]+)", "unit": "μg/dL"},
        "Transferrin Saturation": {"regex": r"Transferrin Saturation[:\s]+([\d.]+)", "unit": "%"},
        "Anti-TPO Antibodies": {"regex": r"Anti-TPO Antibodies[:\s]+([\d.]+)", "unit": "IU/mL"},
        "Rheumatoid Factor": {"regex": r"Rheumatoid Factor[:\s]+([\d.]+)", "unit": "IU/mL"}
    }

    # Extract vitals
    vitals = []
    for param, details in vitals_parameters.items():
        match = re.search(details["regex"], text, re.IGNORECASE)
        if match:
            vitals.append({
                "name": param,
                "value": match.group(1).strip(),
                "unit": details["unit"]
            })

    # Extract complaints as a list
    complaints = re.findall(r"Complaints (.+?)(?:\n|$)", text)
    complaints_list = [complaint.strip() for complaint in complaints]

    # Extract only medication names
    medication_names = re.findall(r"\d\.\s(.*?Tablet)", text)

    # Return the extracted data as a dictionary
    return {
        "doctorName": doctor_name,
        "designation": designations,
        "doctorPhone": doctor_phone,
        "vitals": vitals,
        "complaints": complaints_list,
        "medications": medication_names
    }

if __name__ == '__main__':
    ocr_text = get_ocr(IMAGE_PATH)
    extracted_data = extract_details(ocr_text)

    output_file = "prescription_data.json"
    with open(output_file, "w") as file:
        json.dump(extracted_data, file, indent=4)

    print(f"Data successfully extracted and saved to {output_file}")
