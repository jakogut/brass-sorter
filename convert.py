import sys
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model(sys.argv[1])

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('brass_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
