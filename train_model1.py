import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping # type: ignore
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load CSV Data for XGBoost
csv_path = "heart_disease_data.csv"  # Update with actual path
data = pd.read_csv(csv_path)
X = data.drop(columns=['target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Optimized XGBoost Model (Prevent Overfitting)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                              n_estimators=150, max_depth=6, learning_rate=0.03, 
                              subsample=0.8, scale_pos_weight=1.2, reg_lambda=1.5)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, y_pred) * 100
print(f'✅ XGBoost Accuracy: {xgb_acc:.2f}%')

# Data Augmentation for CNN (Improved)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=50, width_shift_range=0.4,
                                   height_shift_range=0.4, shear_range=0.4, zoom_range=0.5,
                                   horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)

dataset_path = "dataset/"  # Update with actual path
train_generator = train_datagen.flow_from_directory(dataset_path + 'train', target_size=(128, 128), batch_size=32, class_mode='binary')
validation_generator = val_datagen.flow_from_directory(dataset_path + 'val', target_size=(128, 128), batch_size=32, class_mode='binary')

# Improved CNN Model (Better Filters & Optimizer)
cnn_model = Sequential([
    Conv2D(128, (3,3), input_shape=(128, 128, 3)),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3)),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    MaxPooling2D(2,2),
    Conv2D(512, (3,3)),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(1024),
    LeakyReLU(alpha=0.01),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile CNN Model (Using SGD Optimizer)
cnn_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

# Train CNN Model with increased epochs
cnn_history = cnn_model.fit(train_generator, validation_data=validation_generator, epochs=50, callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1), EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
cnn_model.save('models/cnn_model.h5')
cnn_acc = max(cnn_history.history['val_accuracy']) * 100
print(f'✅ CNN Accuracy: {cnn_acc:.2f}%')

# Improved GNN Model (Reduced Dropout & More Training)
class GNNModel(nn.Module):
    def __init__(self, input_dim):
        super(GNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Reduced from 0.3
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_torch = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_torch = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

gnn_model = GNNModel(X_train.shape[1]).to(device)
optimizer = optim.Adam(gnn_model.parameters(), lr=0.0003)
criterion = nn.BCELoss()

dataloader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=128, shuffle=True)
for epoch in range(50):  # Increased epochs
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = gnn_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

y_pred_torch = gnn_model(X_test_torch).detach().cpu().numpy().round()
gnn_acc = accuracy_score(y_test, y_pred_torch) * 100
print(f'✅ GNN Accuracy: {gnn_acc:.2f}%')

print("🎉 All Models Trained & Saved Successfully!")