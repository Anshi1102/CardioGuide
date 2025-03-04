import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

# Load CSV Dataset
csv_path = "heart_disease_data.csv"  # 🛑 Yaha Kaggle dataset ka CSV path lagana hai
df = pd.read_csv(csv_path)

# Handle Missing Values
df.fillna(df.median(), inplace=True)

# Encode Categorical Columns
encoder = LabelEncoder()
df['sex'] = encoder.fit_transform(df['sex'])
df['cp'] = encoder.fit_transform(df['cp'])

# Scale Numeric Features
scaler = StandardScaler()
numerical_cols = ['trestbps', 'chol', 'thalach', 'oldpeak']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Load & Preprocess ECG Images from ALL sub-folders
def preprocess_ecg(image_path):
    try:
        img = load_img(image_path, target_size=(128, 128), color_mode="grayscale")
        img = img_to_array(img) / 255.0  # Normalize
        return img
    except Exception as e:
        print(f"❌ Error loading image: {image_path} - {e}")
        return None

image_folder = "train"  # 🛑 Yaha ECG images ka main folder hai
image_data = []
labels = []

# Loop through all sub-folders
for subdir in os.listdir(image_folder):
    subdir_path = os.path.join(image_folder, subdir)
    if os.path.isdir(subdir_path):  # Check if it's a folder
        for img_file in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, img_file)
            if img_path.lower().endswith((".png", ".jpg", ".jpeg")):  # Check for valid image file
                img = preprocess_ecg(img_path)
                if img is not None:
                    image_data.append(img)

                    # Assign label based on folder name
                    if "Normal" in subdir:
                        labels.append(0)  # 0 for Normal ECG
                    else:
                        labels.append(1)  # 1 for Diseased ECG

# Convert lists to numpy arrays
if image_data:
    image_data = np.array(image_data)
    labels = np.array(labels)
    
    # Save Processed Data
    np.save("X.npy", image_data)
    np.save("y.npy", labels)
    df.to_csv("processed_heart_data.csv", index=False)
    print(f"✅ Data Preprocessing Completed! Loaded {len(image_data)} images.")
else:
    print("❌ No valid images found. Check dataset path and CSV!")