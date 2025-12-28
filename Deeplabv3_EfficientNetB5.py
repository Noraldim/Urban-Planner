import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, GlobalAveragePooling2D, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB5 # <--- CHANGED IMPORT
import numpy as np
import os
import cv2

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = "/content/PASTIS_Fixed"
TARGET_SIZE = (144, 144)
BATCH_SIZE = 8 # EfficientNetB5 is huge, we lower batch size to avoid OOM
EPOCHS = 30
LR = 0.0001
MODEL_SAVE_PATH = "/content/deeplab_efficientnetb5_custom.h5"

# ==========================================
# 2. GLOBAL UTILS (Crucial for Serialization)
# ==========================================
def resize_dynamic(args):
    """Resizes the first tensor to match the height/width of the second tensor."""
    target_tensor, reference_tensor = args
    target_h = tf.shape(reference_tensor)[1]
    target_w = tf.shape(reference_tensor)[2]
    return tf.image.resize(target_tensor, [target_h, target_w])

def scale_up_pixels(x):
    """Scales [0, 1] inputs back to [0, 255] for EfficientNet"""
    return x * 255.0

# ==========================================
# 3. DATA LOADER (Standard)
# ==========================================
class Dataset:
    def __init__(self, images_dir, masks_dir):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, i) for i in self.ids]
        self.masks_fps = [os.path.join(masks_dir, i) for i in self.ids]

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i]); image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        image = cv2.resize(image, TARGET_SIZE)
        mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype('float32'); mask = np.expand_dims(mask, axis=-1)
        image = image / 255.0 # Returns [0, 1]
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
# 4. FIXED MODEL: DEEPLABV3+ (EfficientNetB5)
# ==========================================
def ASPP(inputs):
    # Image Pooling
    y1 = GlobalAveragePooling2D()(inputs)
    y1 = Reshape((1, 1, y1.shape[-1]))(y1)
    y1 = Conv2D(256, 1, padding='same', use_bias=False)(y1)
    y1 = BatchNormalization()(y1); y1 = Activation('relu')(y1)

    # Resize Logic
    y1 = Lambda(resize_dynamic)([y1, inputs])

    y2 = Conv2D(256, 1, padding='same', use_bias=False)(inputs)
    y2 = BatchNormalization()(y2); y2 = Activation('relu')(y2)
    y3 = Conv2D(256, 3, padding='same', dilation_rate=6, use_bias=False)(inputs)
    y3 = BatchNormalization()(y3); y3 = Activation('relu')(y3)
    y4 = Conv2D(256, 3, padding='same', dilation_rate=12, use_bias=False)(inputs)
    y4 = BatchNormalization()(y4); y4 = Activation('relu')(y4)
    y5 = Conv2D(256, 3, padding='same', dilation_rate=18, use_bias=False)(inputs)
    y5 = BatchNormalization()(y5); y5 = Activation('relu')(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Conv2D(256, 1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y); y = Activation('relu')(y)
    return y

def build_deeplab_efficientnet(input_shape):
    inputs = Input(input_shape)

    # FIX: Scale [0, 1] inputs back to [0, 255] for EfficientNet
    scaled_input = Lambda(scale_up_pixels)(inputs)

    # Backbone
    base_model = EfficientNetB5(weights='imagenet', include_top=False, input_tensor=scaled_input)

    # Feature Maps for B5
    # Stride 16 (High Level) -> Block 4a
    image_features = base_model.get_layer('block4a_expand_activation').output
    # Stride 4 (Low Level) -> Block 2a
    low_level_features = base_model.get_layer('block2a_expand_activation').output

    # 1. Process Low Level
    low_level = Conv2D(48, 1, padding='same', use_bias=False)(low_level_features)
    low_level = BatchNormalization()(low_level)
    low_level = Activation('relu')(low_level)

    # 2. Process High Level (ASPP)
    x_a = ASPP(image_features)

    # Resize High Level to match Low Level
    x_a = Lambda(resize_dynamic)([x_a, low_level])

    # 3. Concatenate
    x = Concatenate()([x_a, low_level])

    # Decoder
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)

    x = Conv2D(1, 1, name='prediction')(x)

    # Final Resize
    x = Lambda(lambda img: tf.image.resize(img, (input_shape[0], input_shape[1])))(x)
    outputs = Activation('sigmoid')(x)

    return Model(inputs=inputs, outputs=outputs)

# ==========================================
# 5. TRAIN
# ==========================================
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true); y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def iou_metric(y_true, y_pred):
    y_true_f = K.flatten(y_true); y_pred_f = K.flatten(K.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

print("📂 Loading Data...")
train_dataset = Dataset(os.path.join(DATA_DIR, "images"), os.path.join(DATA_DIR, "masks"))
train_loader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("🏗️ Building DeepLabV3+ (EfficientNetB5)...")
model = build_deeplab_efficientnet((TARGET_SIZE[0], TARGET_SIZE[1], 3))
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='binary_crossentropy', metrics=[dice_coef, iou_metric])

print(f"🚀 Starting Training for EfficientNetB5...")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, mode='max', monitor='iou_metric'),
    tf.keras.callbacks.EarlyStopping(monitor='iou_metric', mode='max', patience=5, restore_best_weights=True)
]

history = model.fit(train_loader, epochs=EPOCHS, callbacks=callbacks)
print(f"✅ EfficientNetB5 Training Finished! Saved to: {MODEL_SAVE_PATH}")
