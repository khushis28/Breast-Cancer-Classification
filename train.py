# import os
# import tensorflow as tf
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_DIR = os.path.join(BASE_DIR, "dataset")
# MODEL_DIR = os.path.join(BASE_DIR, "models")
# os.makedirs(MODEL_DIR, exist_ok=True)

# IMG_SIZE = 224
# BATCH = 16

# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2
# )

# train = datagen.flow_from_directory(
#     DATASET_DIR,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH,
#     class_mode="binary",
#     subset="training"
# )

# val = datagen.flow_from_directory(
#     DATASET_DIR,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH,
#     class_mode="binary",
#     subset="validation"
# )

# base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
# base_model.trainable = False

# model = Sequential([
#     base_model,
#     Flatten(),
#     Dense(256, activation="relu"),
#     Dropout(0.4),
#     Dense(1, activation="sigmoid")
# ])

# model.compile(
#     optimizer="adam",
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )

# model.fit(train, validation_data=val, epochs=10)
# model.save(os.path.join(MODEL_DIR, "model_vgg16.h5"))





import tensorflow as tf
from tensorflow.keras import layers, models, applications
import os

# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "dataset"
EPOCHS = 10

# Load Data from Folders
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary'
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary'
)

# Rescaling Layer
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

def build_transfer_model(base_layer, name):
    base_layer.trainable = False  # Freeze weights
    model = models.Sequential([
        base_layer,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name=name)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize the 3 Models
models_to_train = [
    build_transfer_model(applications.VGG16(input_shape=(224,224,3), include_top=False), "VGG16"),
    build_transfer_model(applications.ResNet50(input_shape=(224,224,3), include_top=False), "ResNet50"),
    build_transfer_model(applications.DenseNet121(input_shape=(224,224,3), include_top=False), "DenseNet121")
]

os.makedirs("models", exist_ok=True)

# Loop to Train and Save
for model in models_to_train:
    print(f"\n🚀 Training {model.name}...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    model.save(f"models/model_{model.name.lower()}.h5")
    print(f"✅ {model.name} saved.")