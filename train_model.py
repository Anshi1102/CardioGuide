import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load dataset with error handling
if not os.path.exists("X.npy") or not os.path.exists("y.npy"):
    raise FileNotFoundError("Dataset files 'X.npy' and 'y.npy' are missing. Please generate or place them in the directory.")

X, y = np.load("X.npy"), np.load("y.npy")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape X_train and X_test for XGBoost (Flatten to 2D)
X_train_xgb = X_train.reshape(X_train.shape[0], -1)
X_test_xgb = X_test.reshape(X_test.shape[0], -1)

# Train XGBoost Model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_xgb, y_train)

# Evaluate XGBoost
xgb_predictions = xgb_model.predict(X_test_xgb)
xgb_accuracy = accuracy_score(y_test, xgb_predictions) * 100
print(f"✅ XGBoost Accuracy: {xgb_accuracy:.2f}%")

# Save XGBoost model
xgb_model.save_model("models/xgboost_model.json")

# Train CNN Model
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Evaluate CNN
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test)
cnn_accuracy *= 100
print(f"✅ CNN Accuracy: {cnn_accuracy:.2f}%")

# Save CNN model
cnn_model.save("models/cnn_model.keras")

# Train GNN Model (Example, update based on your actual GNN architecture)
class GNNModel(tf.keras.Model):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(len(np.unique(y)), activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# Reshape X_train and X_test for GNN (Flatten to 2D)
X_train_gnn = X_train.reshape(X_train.shape[0], -1)
X_test_gnn = X_test.reshape(X_test.shape[0], -1)

gnn_model = GNNModel()
gnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
gnn_model.fit(X_train_gnn, y_train, validation_data=(X_test_gnn, y_test), epochs=10, batch_size=32)

# Evaluate GNN Model
gnn_loss, gnn_accuracy = gnn_model.evaluate(X_test_gnn, y_test)
gnn_accuracy *= 100
print(f"✅ GNN Accuracy: {gnn_accuracy:.2f}%")

# Save GNN Model
gnn_model.save("models/gnn_model.keras")

print("🎉 Model Training Complete & Models Saved!")
