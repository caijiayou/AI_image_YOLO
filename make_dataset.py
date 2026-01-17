import cv2
import os
import numpy as np

# ========= 基本設定 =========
DATASET_DIR = "dataset"
IMG_SIZE = (128, 128)   # KNN 常用尺寸（可改 32x32）
CAMERA_ID = 0

# 建立 0~9 資料夾
for i in range(10):
    os.makedirs(os.path.join(DATASET_DIR, str(i)), exist_ok=True)

# 啟動攝影機
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("❌ 無法開啟攝影機")
    exit()

print("📸 按下 0~9 儲存影像，按 q 離開")

# 計數器（避免覆蓋）
counter = {str(i): len(os.listdir(os.path.join(DATASET_DIR, str(i)))) for i in range(10)}

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 讀取影像失敗")
        break

    # 顯示畫面
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # 按 q 離開
    if key == ord('q'):
        break

    # 按 0~9 儲存
    if ord('0') <= key <= ord('9'):
        label = chr(key)

        # 1️⃣ 轉灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2️⃣ 縮小尺寸
        resized = cv2.resize(gray, IMG_SIZE)

        # 3️⃣ 正規化 (0~1)
        normalized = resized / 255.0

        # 4️⃣ 轉回 uint8 儲存（KNN 訓練前再轉 float 也可以）
        save_img = (normalized * 255).astype(np.uint8)

        counter[label] += 1
        filename = f"{label}_{counter[label]:03d}.png"
        filepath = os.path.join(DATASET_DIR, label, filename)

        cv2.imwrite(filepath, save_img)
        print(f"✅ 儲存：{filepath}")

cap.release()
cv2.destroyAllWindows()
