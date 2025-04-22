from flask import Flask, request, jsonify
from flask_cors import CORS  # Impor flask_cors
from joblib import load
import numpy as np
import os

# Memuat pipeline model
model_path = os.path.join(os.path.dirname(__file__), "../deployment/crop_recommendation_pipeline.pkl")
model = load(model_path)

# Inisialisasi aplikasi Flask
app = Flask(__name__)
CORS(app)  # Mengaktifkan CORS

@app.route('/')
def home():
    return "Selamat datang di aplikasi model rekomendasi tanaman!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data input dari request JSON
        data = request.json
        # Validasi input untuk memastikan semua nilai ada
        required_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for feature in required_features:
            if feature not in data:
                return jsonify({'status': 'error', 'message': f'Missing feature: {feature}'})
        
        features = np.array([
            data['N'], data['P'], data['K'],
            data['temperature'], data['humidity'],
            data['ph'], data['rainfall']
        ]).reshape(1, -1)

        # Prediksi menggunakan model
        prediction = model.predict(features)[0]
        
        return jsonify({
            'status': 'success',
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
