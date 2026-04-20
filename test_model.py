import os
import joblib
import numpy as np

def test_model_exists():
    """Kiểm tra xem file mô hình đã được tạo thành công chưa"""
    assert os.path.exists("models/model.pkl"), "File mô hình không tồn tại. Hãy chạy train.py trước."

def test_model_prediction():
    """Kiểm tra xem mô hình có thể đưa ra dự đoán hợp lệ không"""
    model = joblib.load("models/model.pkl")
    # Dữ liệu giả lập với 8 features tương ứng với Pima Indians Diabetes dataset
    dummy_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
    prediction = model.predict(dummy_data)
    
    assert len(prediction) == 1, "Mô hình phải trả về 1 kết quả dự đoán"
    assert prediction[0] in [0, 1], "Kết quả dự đoán phải là 0 hoặc 1"
