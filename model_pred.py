import cv2
import numpy as np
import joblib

# ========= 載入模型與 scaler =========
knn = joblib.load("knn_digit_model.pkl")

IMG_SIZE = (128, 128)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, IMG_SIZE)
    img_norm = img.astype(np.float32) / 255.0

    X_input = img_norm.flatten().reshape(1, -1)
    pred = knn.predict(X_input)

    # 在影像上顯示預測結果
    cv2.putText(frame, f"Pred: {pred[0]}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
