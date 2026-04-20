import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train():
    seed = 42
    np.random.seed(seed)
    
    print("[*] Đang tải dữ liệu...")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    df = pd.read_csv(url, names=names)
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    print("[*] Đang huấn luyện mô hình...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)
    clf.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[*] Độ chính xác (Accuracy): {acc:.4f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/model.pkl")
    print("[*] Đã lưu mô hình tại models/model.pkl")

if __name__ == "__main__":
    train()
