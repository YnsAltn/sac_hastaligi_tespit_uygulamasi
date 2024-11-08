import tensorflow as tf

# Eğitilmiş modelinizi yükleyin
model = tf.keras.models.load_model('sac_hastaliklari2_model.h5')

# TFLite modeline dönüştürme
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# TFLite modelini kaydedin
with open('sac_hastaliklari2_model.tflite', 'wb') as f:
    f.write(tflite_model)
