from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Görüntü boyutları
IMAGE_SIZE = [224, 224]

# Veri seti yolları
train_path = r'C:/Users/yunus/Desktop/Bitirme/AI_Sac_Hastaligi_Teshis_Uygulamasi/data/train'
test_path = r'C:/Users/yunus/Desktop/Bitirme/AI_Sac_Hastaligi_Teshis_Uygulamasi/data/test'
val_path = r'C:/Users/yunus/Desktop/Bitirme/AI_Sac_Hastaligi_Teshis_Uygulamasi/data/val'

# Modelin giriş katmanı
inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# Konvolüsyonel katmanlar
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Düzleştirme ve tam bağlantılı katmanlar
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)  # 10 sınıf için softmax

# Modelin tanımı
model = Model(inputs=inputs, outputs=outputs)

# Modelin derlenmesi
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=["accuracy"]
)

# Modelin yapısını göster
model.summary()

# Görüntü veri artırma (rescale)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim, test ve doğrulama setlerinin oluşturulması
training_set = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

val_set = val_datagen.flow_from_directory(
    directory=val_path,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

# Modelin eğitilmesi
history = model.fit(
    training_set,
    validation_data=val_set,
    epochs=20,
    steps_per_epoch=len(training_set),
    validation_steps=len(val_set)  # val_set'te validation_steps kullan
)

# Modelin kaydedilmesi
model.save("sac_hastaliklari2_model.h5")

# Eğitim tarihinin kaydedilmesi
np.save('sac_hastaliklari2_history.npy', history.history)

# Test seti ile modelin değerlendirilmesi
eval_result = model.evaluate(test_set)
test_loss, test_acc = eval_result[0], eval_result[1]
print(f'Test Accuracy: {test_acc * 100:.2f}%')
print(f'Test Loss: {test_loss:.4f}')

# Eğitim ve doğrulama kaybının görselleştirilmesi
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Eğitim ve doğrulama doğruluğunun görselleştirilmesi
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')  # 'acc' yerine 'accuracy' kullanılmalı
plt.plot(history.history['val_accuracy'], label='Val Accuracy')  # 'val_acc' yerine 'val_accuracy' kullanılmalı
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
