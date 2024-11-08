import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Modelin kaydedildiği dosya yolu
model_path = 'C:/Users/yunus/Desktop/Bitirme/AI_Sac_Hastaligi_Teshis_Uygulamasi/sac_hastaliklari2_model.h5'

# Sınıf isimleri
class_names = ['Alopecia Areata', 'Contact Dermatitis', 'Folliculitis', 'Head Lice', 'Lichen Planus',
               'Male Pattern Baldness', 'Psoriasis', 'Seborrheic Dermatitis', 'Telogen Effluvium', 'Tinea Capitis']

# Görüntü dosya yolunu belirleyin
image_path = "C:/Users/yunus/Desktop/sedef.jpeg"
# Görüntüyü hazırlayan fonksiyon
IMG_SIZE = (224, 224)


def prepare_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)  # Model giriş boyutuna uygun hale getirin
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Modelin beklediği 4 boyutlu format (1, 224, 224, 3)
    img_array /= 255.0  # Normalizasyon
    return img_array


# Modeli yükleyin
model = tf.keras.models.load_model(model_path)


# Görüntüyü tahmin için modelde kullanma fonksiyonu
def predict_image_class(img_path):
    img_array = prepare_image(img_path)
    predictions = model.predict(img_array)

    # En yüksek olasılığa sahip sınıfı bul
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class]
    predicted_probability = predictions[0][predicted_class]

    # Olasılıkları ve en yüksek olasılığı yazdırın
    print("\nOlasılıklar:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {predictions[0][i] * 100:.2f}%")

    print(f"\nTahmin Edilen Sınıf: {predicted_class_name} - Olasılık: {predicted_probability * 100:.2f}%")

    # Eşik değerini kontrol et
    if predicted_probability < 0.7:  # 0.7 bir eşik değeri
        print("Belirsiz sonuç, tekrar kontrol edilmesi önerilir.")
    return predicted_class


# Belirli bir resim üzerinde tahmin yapma
print(f"\nTahmin yapılıyor: {os.path.basename(image_path)}")
predicted_disease = predict_image_class(image_path)
print(f"Tahmin Edilen Hastalık: {class_names[predicted_disease]}")
