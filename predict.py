import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# =========================
# CONFIG
# =========================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "carbon_model.h5")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
IMG_SIZE = (128, 128)

# =========================
# LOAD MODEL & ENCODER
# =========================
print("📥 Loading model...")
model = load_model(
    MODEL_PATH,
    compile=False  # ✅ fixes the keras.metrics.mse issue
)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    # Load and preprocess image
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    co2e, carbon_credit = prediction[0]

    return co2e, carbon_credit

# =========================
# TEST PREDICTION
# =========================
if __name__ == "__main__":
    test_image = "images/image-maize2.jpeg"  # 🔁 change to any test image
    co2e, carbon_credit = predict_image(test_image)

    print(f"🌱 Prediction for {test_image}:")
    print(f"   - CO2e (kg): {co2e:.2f}")
    print(f"   - Carbon Credit: {carbon_credit:.2f}")
