import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16  # VGG16 modelini içe aktar

# Define paths to image directories
train_dir = r"C:/Users/yunus/Desktop/Bitirme/AI_Sac_Hastaligi_Teshis_Uygulamasi/data/train"
val_dir = r"C:/Users/yunus/Desktop/Bitirme/AI_Sac_Hastaligi_Teshis_Uygulamasi/data/val"
test_dir = r"C:/Users/yunus/Desktop/Bitirme/AI_Sac_Hastaligi_Teshis_Uygulamasi/data/test"

# Image dimensions and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Create image datasets
train_dataset = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

validation_dataset = image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_dataset = image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Extract class names
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)
print("Class names:", class_names)

# Load VGG16 model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Freeze the base model
base_model.trainable = False

# Define the new model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Global Average Pooling layer
    layers.Dense(256, activation='relu'),  # New dense layer
    layers.Dropout(0.5),  # Dropout for regularization
    layers.Dense(num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Set early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train model with 20 epochs
history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=10,
                    callbacks=[early_stopping])

# Save the model
model.save("sac_hastaligi_model.h5")

# Evaluate model on test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.tight_layout()
plt.show()

# Prediction function for new images
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)  # Resize image
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
    img_array /= 255.0  # Normalize
    return img_array

def predict_image_class(img_path):
    img_array = prepare_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return class_names[predicted_class[0]]

# Test with a new image
image_path = "C:/Users/yunus/Desktop/Bitirme/AI_Sac_Hastaligi_Teshhis_Uygulamasi/data/test/Head Lice/head_lice_0148.jpg"
predicted_disease = predict_image_class(image_path)
print("Predicted Disease:", predicted_disease)
