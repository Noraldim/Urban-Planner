import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

# ==========================================
# 1. SETUP
# ==========================================
#MODEL_PATH = "/content/deeplab_efficientnetb5_custom.h5"
#MODEL_PATH = "/content/deeplab_resnet50_custom.h5"
MODEL_PATH = "/content/deeplab_resnet101_custom.h5"
DATA_DIR = "/content/PASTIS_Fixed"
TARGET_SIZE = (144, 144)

# Load Weights
print("📂 Loading EfficientNetB5 Model...")
model = build_deeplab_101((TARGET_SIZE[0], TARGET_SIZE[1], 3))
model.load_weights(MODEL_PATH)
print("✅ Model Loaded!")

# ==========================================
# 2. EVALUATION LOOP
# ==========================================
images_dir = os.path.join(DATA_DIR, "images")
masks_dir = os.path.join(DATA_DIR, "masks")
ids = os.listdir(images_dir)

iou_scores = []
dice_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []

print(f"🚀 Evaluating on {len(ids)} images...")

for i, filename in enumerate(ids):
    mask_path = os.path.join(masks_dir, filename)
    if not os.path.exists(mask_path): continue

    # A. Process Image
    img = cv2.imread(os.path.join(images_dir, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE)
    img_input = np.expand_dims(img / 255.0, axis=0) # [0, 1] input (Model handles scaling)

    # B. Process Mask
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    y_true = (mask > 0).astype(np.uint8).flatten()

    # C. Predict
    pred = model.predict(img_input, verbose=0)[0, :, :, 0]
    y_pred = (pred > 0.5).astype(np.uint8).flatten()

    # D. Calculate Metrics
    iou_scores.append(jaccard_score(y_true, y_pred, average='binary', zero_division=1))
    dice_scores.append(f1_score(y_true, y_pred, average='binary', zero_division=1))
    accuracy_scores.append(accuracy_score(y_true, y_pred))
    precision_scores.append(precision_score(y_true, y_pred, average='binary', zero_division=1))
    recall_scores.append(recall_score(y_true, y_pred, average='binary', zero_division=1))

    # Progress
    if i % 500 == 0:
        print(f"Processing image {i}/{len(ids)}...")

    # Visualize 3 random samples
    if i < 3:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title("Original (EffNetB5)")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pred > 0.5, cmap='plasma') # Using plasma for visual distinction
        plt.title(f"Prediction (IoU: {iou_scores[-1]:.2f})")
        plt.axis('off')
        plt.show()

# ==========================================
# 3. FINAL REPORT
# ==========================================
print("\n" + "="*40)
print("📊 EFFICIENTNETB5 FINAL REPORT")
print("="*40)
print(f"Mean IoU (Jaccard):   {np.mean(iou_scores):.4f}")
print(f"Mean Dice (F1 Score): {np.mean(dice_scores):.4f}")
print(f"Mean Pixel Accuracy:  {np.mean(accuracy_scores):.4f}")
print(f"Mean Precision:       {np.mean(precision_scores):.4f}")
print(f"Mean Recall:          {np.mean(recall_scores):.4f}")
print("="*40)
