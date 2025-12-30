# 1. FIX KERAS BUG (Just in case)
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import types

if not hasattr(keras.utils, 'generic_utils'):
    keras.utils.generic_utils = types.ModuleType('generic_utils')
    keras.utils.generic_utils.get_custom_objects = keras.utils.get_custom_objects
    keras.utils.generic_utils.CustomObjectScope = keras.utils.custom_object_scope

import segmentation_models as sm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

# ==========================================
# 2. CONFIGURATION
# ==========================================
# --- CHANGED TO RESNET101 ---
BACKBONE = 'resnet101'

# ⚠️ CHECK THIS PATH: Make sure it points to your ResNet101 file
MODEL_PATH = "/content/drive/MyDrive/unet_resnet101_customxx.h5"

DATA_DIR = "/content/PASTIS_Fixed"
TARGET_SIZE = (160, 160) # Must stay 160x160 for ResNet101

# ==========================================
# 3. LOAD MODEL
# ==========================================
print(f"📂 Loading U-Net ({BACKBONE})...")

# Build architecture
model = sm.Unet(BACKBONE, classes=1, activation='sigmoid')

# Load weights
try:
    model.load_weights(MODEL_PATH)
    print("✅ Model Weights Loaded!")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    print("👉 Check if the file path is correct!")

# Get ResNet101 specific preprocessing
preprocess_input = sm.get_preprocessing(BACKBONE)

# ==========================================
# 4. EVALUATION LOOP (ALL METRICS)
# ==========================================
images_dir = os.path.join(DATA_DIR, "images")
masks_dir = os.path.join(DATA_DIR, "masks")
ids = os.listdir(images_dir)

# Initialize Lists
iou_scores = []
dice_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []

print(f"🚀 Evaluating on {len(ids)} images... (This may take a moment)")

for i, filename in enumerate(ids):
    mask_path = os.path.join(masks_dir, filename)
    if not os.path.exists(mask_path): continue

    # A. Load & Preprocess Image
    img_raw = cv2.imread(os.path.join(images_dir, filename))
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_raw = cv2.resize(img_raw, TARGET_SIZE)

    # ResNet101 Preprocessing is crucial!
    img_input = preprocess_input(img_raw)
    img_input = np.expand_dims(img_input, axis=0)

    # B. Load & Preprocess Mask
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    y_true = (mask > 0).astype(np.uint8).flatten() # Flatten for sklearn metrics

    # C. Predict
    pred = model.predict(img_input, verbose=0)[0, :, :, 0]
    y_pred = (pred > 0.5).astype(np.uint8).flatten()

    # D. Calculate Metrics
    # 1. IoU (Intersection over Union)
    iou_scores.append(jaccard_score(y_true, y_pred, average='binary', zero_division=1))

    # 2. Dice / F1 Score
    dice_scores.append(f1_score(y_true, y_pred, average='binary', zero_division=1))

    # 3. Accuracy
    accuracy_scores.append(accuracy_score(y_true, y_pred))

    # 4. Precision
    precision_scores.append(precision_score(y_true, y_pred, average='binary', zero_division=1))

    # 5. Recall
    recall_scores.append(recall_score(y_true, y_pred, average='binary', zero_division=1))

    # Progress Log (every 500 images)
    if i % 500 == 0 and i > 0:
        print(f"   Processed {i}/{len(ids)}...")

    # Visualize first 3 results
    if i < 3:
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1); plt.imshow(img_raw.astype('uint8')); plt.title("Original")
        plt.subplot(1, 3, 2); plt.imshow(mask, cmap='gray'); plt.title("Ground Truth")
        plt.subplot(1, 3, 3); plt.imshow(pred > 0.5, cmap='jet'); plt.title(f"Prediction (IoU: {iou_scores[-1]:.2f})")
        plt.show()

# ==========================================
# 5. FINAL COMPREHENSIVE REPORT
# ==========================================
print("\n" + "="*40)
print(f"📊 FINAL REPORT: U-Net ({BACKBONE})")
print("="*40)
print(f"Mean IoU (Jaccard):   {np.mean(iou_scores):.4f}")
print(f"Mean Dice (F1 Score): {np.mean(dice_scores):.4f}")
print(f"Mean Pixel Accuracy:  {np.mean(accuracy_scores):.4f}")
print(f"Mean Precision:       {np.mean(precision_scores):.4f}")
print(f"Mean Recall:          {np.mean(recall_scores):.4f}")
print("="*40)
