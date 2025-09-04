import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

# =========================
# CONFIG
# =========================
DATASET_PATH = "data.csv"
IMAGE_DIR = "images"
MODEL_DIR = "models"
IMG_SIZE = (128, 128)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(DATASET_PATH)

images = []
for img_name in df["image"]:
    img_path = os.path.join(IMAGE_DIR, img_name)
    if os.path.exists(img_path):
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
    else:
        print(f"⚠️ Warning: {img_path} not found. Using blank image.")
        images.append(np.zeros((*IMG_SIZE, 3)))

images = np.array(images)

# Encode crop type (optional, in case you want to use later)
label_encoder = LabelEncoder()
crop_labels = label_encoder.fit_transform(df["crop_type"])

# Targets: CO2e + Carbon Credits
targets = df[["co2e", "carbon_credit"]].values

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    images, targets, test_size=0.2, random_state=42
)

# =========================
# BUILD CNN MODEL
# =========================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='linear')  # 2 outputs: co2e & carbon_credit
])

# =========================
# COMPILE MODEL
# =========================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=MeanSquaredError(),
    metrics=[MeanAbsoluteError()]
)

# =========================
# TRAIN MODEL
# =========================
print("🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=4,
    verbose=1
)

# =========================
# SAVE MODEL & ENCODER
# =========================
model.save(os.path.join(MODEL_DIR, "carbon_model.h5"))

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Training complete. Model saved in /models/")
