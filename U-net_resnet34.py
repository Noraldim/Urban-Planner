# ================= MODEL SETUP =================
# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define Backbone
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# Preprocess for ResNet
X_train_pre = preprocess_input(X_train * 255) # Re-scale for ResNet logic
X_test_pre  = preprocess_input(X_test * 255)

# Define Model (BINARY MODE)
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid')

# Define Metrics & Loss
# For Binary, we use 'binary_crossentropy' + 'Jaccard' (IoU)
model.compile(
    optimizer='adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score]
)

print(model.summary())

def jaccard_coef(y_true, y_pred, smooth=1):
    # --- THE FIX IS HERE ---
    # We explicitly cast y_true to float32 to match y_pred
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# 2. Define Jaccard Loss
def jaccard_loss(y_true, y_pred):
    return 1 - jaccard_coef(y_true, y_pred)

# 3. Define Combined Loss (Binary Crossentropy + Jaccard)
def bce_jaccard_loss(y_true, y_pred):
    # Ensure y_true is float32 for binary_crossentropy as well
    y_true_casted = K.cast(y_true, 'float32')
    bce = tf.keras.losses.binary_crossentropy(y_true_casted, y_pred)
    jaccard = jaccard_loss(y_true, y_pred)
    return bce + jaccard

print("✅ Custom Loss Functions defined (with Float32 casting).")

# ====================================================
# RE-COMPILE & TRAIN
# ====================================================

model.compile(
    optimizer='adam',
    loss=bce_jaccard_loss,
    metrics=[jaccard_coef, 'accuracy']
)

print("🚀 Restarting Training (The Planner)...")

history = model.fit(
    X_train_pre,
    y_train,
    batch_size=32,
    epochs=30,
    verbose=1,
    validation_data=(X_test_pre, y_test)
)

# ================= SAVE MODEL =================
save_path = '/content/drive/MyDrive/saved_models'
os.makedirs(save_path, exist_ok=True)
model.save(f'{save_path}/pastis_planner_model.h5')
print("✅ Model Saved!")
