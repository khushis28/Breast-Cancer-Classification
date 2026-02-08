# import tensorflow as tf
# import numpy as np
# import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt

# IMG_SIZE = 224
# MODEL_PATH = "models/model_vgg16.h5"
# DATASET_DIR = "dataset"

# model = tf.keras.models.load_model(MODEL_PATH)

# datagen = ImageDataGenerator(rescale=1./255)

# test = datagen.flow_from_directory(
#     DATASET_DIR,
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=1,
#     class_mode="binary",
#     shuffle=False
# )

# preds = model.predict(test)
# y_pred = (preds > 0.5).astype(int)

# print("📊 Classification Report")
# print(classification_report(test.classes, y_pred, target_names=["Benign", "Malignant"]))

# cm = confusion_matrix(test.classes, y_pred)
# plt.imshow(cm, cmap="Blues")
# plt.title("Confusion Matrix")
# plt.colorbar()
# plt.show()





import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Select which model to evaluate: model_vgg16.h5, model_resnet50.h5, or model_densenet121.h5
MODEL_NAME = "model_vgg16.h5" 
MODEL_PATH = f"models/{MODEL_NAME}"

# Load Data (Shuffle=False is CRITICAL for confusion matrix)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False, 
    label_mode='binary'
)

# Rescale
normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Get True Labels
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# Load and Predict
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    preds = model.predict(test_ds)
    y_pred = (preds > 0.5).astype(int)

    # Classification Report
    print(f"\n📊 Performance Report for {MODEL_NAME}")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))

    # Confusion Matrix Image
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Benign", "Malignant"], 
                yticklabels=["Benign", "Malignant"])
    
    plt.title(f"Confusion Matrix: {MODEL_NAME}")
    plt.ylabel('Actual Label')
    plt.xlabel('AI Predicted Label')
    plt.savefig(f"models/cm_{MODEL_NAME.replace('.h5', '.png')}")
    plt.show()
else:
    print("❌ Model not found. Run train_comparison.py first.")