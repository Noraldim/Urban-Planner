# 1. 
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
BACKBONE = 'resnet101'
BATCH_SIZE = 16
EPOCHS = 30
LR = 0.0001
# --- FIX IS HERE ---
TARGET_SIZE = (160, 160) # Changed from 144 to 160 (Must be divisible by 32)
MODEL_SAVE_PATH = f'/content/drive/MyDrive/unet_{BACKBONE}_customxx.h5'

# Define Preprocessor specific to ResNet101
preprocess_input = sm.get_preprocessing(BACKBONE)

# ==========================================
# 3. DATA LOADER (Binary Black & White Fix)
# ==========================================
class Dataset:
    def __init__(self, images_dir, masks_dir, preprocessing=None):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, i) for i in self.ids]
        self.masks_fps = [os.path.join(masks_dir, i) for i in self.ids]
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # Read Image
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, TARGET_SIZE)

        # Read Mask (Force Gray)
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        # --- COLOR FIX ---
        # Convert any color > 0 to exactly 1.0 (White)
        # This ensures the model learns strictly Black (0) vs White (1)
        mask = (mask > 0).astype('float32')
        mask = np.expand_dims(mask, axis=-1)

        # Apply Preprocessing (Crucial for ResNet101)
        if self.preprocessing:
            image = self.preprocessing(image)

        return image, mask

    def __len__(self): return len(self.ids)

class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size=8, shuffle=False):
        self.dataset = dataset; self.batch_size = batch_size; self.shuffle = shuffle
        self.indexes = np.arange(len(dataset)); self.on_epoch_end()
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        data = [self.dataset[i] for i in indexes]
        batch = [np.stack(s, axis=0) for s in zip(*data)]
        return batch
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indexes)
    def __len__(self): return len(self.dataset) // self.batch_size

# ==========================================
# 4. PREPARE DATA
# ==========================================
print("📂 Preparing Data...")
train_dataset = Dataset(
    os.path.join(DATA_DIR, "images"),
    os.path.join(DATA_DIR, "masks"),
    preprocessing=preprocess_input
)
train_loader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 5. BUILD MODEL
# ==========================================
print(f"🏗️ Building U-Net with {BACKBONE} backbone...")

model = sm.Unet(
    BACKBONE,
    encoder_weights='imagenet',
    classes=1,
    activation='sigmoid' # Ensures output is 0.0 to 1.0
)

# Robust Loss Function (BCE + Jaccard)
total_loss = sm.losses.bce_jaccard_loss
metrics = [sm.metrics.iou_score, sm.metrics.f1_score]

model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=total_loss, metrics=metrics)

# ==========================================
# 6. TRAIN
# ==========================================
print("🚀 Starting Training...")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, mode='max', monitor='iou_score'),
    tf.keras.callbacks.EarlyStopping(monitor='iou_score', patience=5, mode='max', restore_best_weights=True)
]

history = model.fit(
    train_loader,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f"✅ Training Finished! Model saved to {MODEL_SAVE_PATH}")
