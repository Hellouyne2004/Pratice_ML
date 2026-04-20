from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

model_path = "models/model.pkl"
model = None

# Tải mô hình khi khởi động ứng dụng
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("[*] Đã tải mô hình thành công.")
else:
    print("[!] Cảnh báo: Không tìm thấy file mô hình. Vui lòng chạy train.py.")

@app.route("/")
def home():
    return jsonify({"message": "ML Model API đang hoạt động!", "status": "running"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Mô hình chưa được tải"}), 500
    
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        
        return jsonify({
            "prediction": int(prediction[0]),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
