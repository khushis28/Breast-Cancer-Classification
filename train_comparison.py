import tensorflow as tf
from tensorflow.keras import layers, models, applications
import os

# ================= CONFIG =================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "dataset"
EPOCHS = 10

# ================= LOAD DATA =================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# ================= NORMALIZATION =================
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ================= MODEL =================
base_model = applications.VGG16(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ================= TRAIN =================
print("🚀 Training Started...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ================= SAVE MODEL =================
os.makedirs("models", exist_ok=True)

model.save("models/model_vgg16.h5")

print("✅ Model Saved Successfully")