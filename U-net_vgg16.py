# 1. FIX KERAS BUG (Run this first!)
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import tensorflow as tf

# Patch generic_utils for segmentation_models
if not hasattr(keras.utils, 'generic_utils'):
    import types
    keras.utils.generic_utils = types.ModuleType('generic_utils')
    keras.utils.generic_utils.get_custom_objects = keras.utils.get_custom_objects
    keras.utils.generic_utils.CustomObjectScope = keras.utils.custom_object_scope

import segmentation_models as sm
import numpy as np
import cv2

# ==========================================
# 2. CONFIGURATION
# ==========================================
DATA_DIR = "/content/PASTIS_Fixed"
BACKBONE = 'vgg16'  # <--- CHANGED TO VGG16
BATCH_SIZE = 16     # VGG16 is heavy; if you get OOM (Out Of Memory), reduce to 8
EPOCHS = 30
LR = 0.0001
# Keep 160x160 (VGG16 also requires divisibility by 32)
TARGET_SIZE = (160, 160) 
MODEL_SAVE_PATH = f'/content/drive/MyDrive/saved_models/unet_{BACKBONE}_customxx.h5'

# Define Preprocessor specific to VGG16
preprocess_input = sm.get_preprocessing(BACKBONE)
# ... (Keep Section 1 and 2 exactly as they are) ...

# ==========================================
# 3. FAST DATA LOADER (RAM CACHED)
# ==========================================
def load_data_into_ram(images_dir, masks_dir, target_size, preprocessing=None):
    image_ids = sorted(os.listdir(images_dir))
    
    # Pre-allocate arrays for speed (Batch, Height, Width, Channels)
    # 3 channels for image, 1 channel for mask
    num_samples = len(image_ids)
    X = np.zeros((num_samples, target_size[0], target_size[1], 3), dtype=np.float32)
    Y = np.zeros((num_samples, target_size[0], target_size[1], 1), dtype=np.float32)
    
    print(f"📂 Loading {num_samples} images into RAM...")
    
    for i, img_id in enumerate(image_ids):
        # Paths
        img_path = os.path.join(images_dir, img_id)
        mask_path = os.path.join(masks_dir, img_id)
        
        # Read & Resize Image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        
        # Read & Resize Mask
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Preprocessing (VGG16 specific)
        if preprocessing:
            img = preprocessing(img)
            
        # Binary Mask Fix
        mask = (mask > 0).astype('float32')
        mask = np.expand_dims(mask, axis=-1)
        
        # Store in array
        X[i] = img
        Y[i] = mask
        
        if i % 500 == 0 and i > 0:
            print(f"   Loaded {i} / {num_samples}...")
            
    print("✅ All data loaded into RAM!")
    return X, Y

# ==========================================
# 4. PREPARE DATA
# ==========================================
# We load ALL data into X_train and Y_train variables
X_train, Y_train = load_data_into_ram(
    os.path.join(DATA_DIR, "images"), 
    os.path.join(DATA_DIR, "masks"), 
    target_size=TARGET_SIZE,
    preprocessing=preprocess_input
)

# ==========================================
# 5. BUILD MODEL (Same as before)
# ==========================================
print(f"🏗️ Building U-Net with {BACKBONE} backbone...")

model = sm.Unet(
    BACKBONE, 
    encoder_weights='imagenet', 
    classes=1, 
    activation='sigmoid'
)

total_loss = sm.losses.bce_jaccard_loss
metrics = [sm.metrics.iou_score, sm.metrics.f1_score]

model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=total_loss, metrics=metrics)

# ==========================================
# 6. TRAIN (OPTIMIZED)
# ==========================================
# INCREASE BATCH SIZE FOR A100
# 16 is too small. A100 can handle 64, 128, or even 256 easily.
OPTIMIZED_BATCH_SIZE = 64 

print(f"🚀 Starting Training on A100 (Batch Size: {OPTIMIZED_BATCH_SIZE})...")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, mode='max', monitor='iou_score'),
    tf.keras.callbacks.EarlyStopping(monitor='iou_score', patience=5, mode='max', restore_best_weights=True)
]

# Note: We pass X_train and Y_train directly, NOT a loader object
history = model.fit(
    x=X_train,
    y=Y_train,
    batch_size=OPTIMIZED_BATCH_SIZE, 
    epochs=EPOCHS, 
    validation_split=0.1, # Optional: Splits 10% for validation automatically
    callbacks=callbacks,
    verbose=1
)

print(f"✅ Training Finished! Model saved to {MODEL_SAVE_PATH}")
