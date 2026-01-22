from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # for CORS support
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../model', 'loan_model.pkl')
print(f"📦 Loading model from: {MODEL_PATH}")

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise RuntimeError("🚫 Model file not found. Make sure the path is correct.")

app = Flask(__name__, static_folder='../frontend/build', static_url_path='')
CORS(app, origins=["http://localhost:5173"]) 

def preprocess_input(data):
    try:
        return [
            1 if data['Gender'] == 'Male' else 0,
            1 if data['Married'] == 'Yes' else 0,
            {'0': 0, '1': 1, '2': 2, '3+': 3}.get(str(data['Dependents']), 0),
            1 if data['Education'] == 'Graduate' else 0,
            1 if data['Self_Employed'] == 'Yes' else 0,
            float(data['ApplicantIncome']),
            float(data['CoapplicantIncome']),
            float(data['LoanAmount']) / 1000,
            float(data['Loan_Amount_Term']),
            1.0 if data['Credit_History'] in ['Y', '1', 1] else 0.0,
            {'Urban': 2, 'Semiurban': 1, 'Rural': 0}[data['Property_Area']]
        ]
    except Exception as e:
        raise ValueError(f"Invalid input format: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("📥 Received data:", data)

        if not data:
            return jsonify({'error': 'Missing input data'}), 400

        features = preprocess_input(data)
        prediction = model.predict([np.array(features)])

        result = "Loan Approved" if prediction[0] == 'Y' else "Loan Rejected"

        return jsonify({'prediction': result})

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    target_path = os.path.join(app.static_folder, path)
    if path != "" and os.path.exists(target_path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
