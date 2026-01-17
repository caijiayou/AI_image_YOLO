import cv2
import numpy as np
import joblib

# ========= 載入模型 =========
knn = joblib.load("knn_digit_model.pkl")

IMG_SIZE = (28, 28)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, IMG_SIZE)
    img = img / 255.0

    X_input = img.flatten().reshape(1, -1)
    pred = knn.predict(X_input)

    print("🎯 預測結果:", pred[0])

cap.release()
cv2.destroyAllWindows()
