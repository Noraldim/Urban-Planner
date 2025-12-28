import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, UpSampling2D, GlobalAveragePooling2D, Reshape, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet101 # <--- CHANGED IMPORT
import numpy as np
import os
import cv2

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = "/content/PASTIS_Fixed"
TARGET_SIZE = (144, 144)
BATCH_SIZE = 16
EPOCHS = 30
LR = 0.0001
MODEL_SAVE_PATH = "/content/deeplab_resnet101_custom.h5" # <--- NEW FILENAME

# ==========================================
# 2. DATA LOADER (Same as before)
# ==========================================
class Dataset:
    def __init__(self, images_dir, masks_dir):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, i) for i in self.ids]
        self.masks_fps = [os.path.join(masks_dir, i) for i in self.ids]

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        image = cv2.resize(image, TARGET_SIZE)
        mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype('float32')
        mask = np.expand_dims(mask, axis=-1)
        image = image / 255.0
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
# 3. MODEL: DEEPLABV3+ (ResNet101)
# ==========================================
def ASPP(inputs):
    shape = inputs.shape
    y1 = GlobalAveragePooling2D()(inputs)
    y1 = Reshape((1, 1, y1.shape[-1]))(y1)
    y1 = Conv2D(256, 1, padding='same', use_bias=False)(y1)
    y1 = BatchNormalization()(y1); y1 = Activation('relu')(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)
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

def build_deeplab_101(input_shape):
    inputs = Input(input_shape)
    # Using ResNet101 Backbone
    base_model = ResNet101(weights='imagenet', include_top=False, input_tensor=inputs)

    # --- KEY CHANGE FOR RESNET101 ---
    # High-Level Features (Stride 16): 'conv4_block23_out' (instead of block6)
    image_features = base_model.get_layer('conv4_block23_out').output
    # Low-Level Features (Stride 4): Same as ResNet50
    low_level_features = base_model.get_layer('conv2_block3_out').output

    # ASPP
    x_a = ASPP(image_features)
    x_a = UpSampling2D((4, 4), interpolation='bilinear')(x_a)

    # Decoder
    low_level_features = Conv2D(48, 1, padding='same', use_bias=False)(low_level_features)
    low_level_features = BatchNormalization()(low_level_features)
    low_level_features = Activation('relu')(low_level_features)

    x = Concatenate()([x_a, low_level_features])
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)

    x = Conv2D(1, 1, name='prediction')(x)
    x = Lambda(lambda img: tf.image.resize(img, (input_shape[0], input_shape[1])))(x)
    outputs = Activation('sigmoid')(x)

    return Model(inputs=inputs, outputs=outputs)

# ==========================================
# 4. TRAIN
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

print("🏗️ Building DeepLabV3+ (ResNet101)...")
model = build_deeplab_101((TARGET_SIZE[0], TARGET_SIZE[1], 3))
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='binary_crossentropy', metrics=[dice_coef, iou_metric])

print(f"🚀 Starting Training for ResNet101...")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, mode='max', monitor='iou_metric'),
    tf.keras.callbacks.EarlyStopping(monitor='iou_metric', mode='max', patience=5, restore_best_weights=True)
]

history = model.fit(train_loader, epochs=EPOCHS, callbacks=callbacks)
print(f"✅ ResNet101 Training Finished! Saved to: {MODEL_SAVE_PATH}")
