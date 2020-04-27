'''
Loads a trained model, and classifies an image

argv[1]: path to tflite model to load
argv[2]: path to image to classify
'''

full_tf = True

try:
    import tensorflow as tf
except (ImportError, ModuleNotFoundError):
    import tflite_runtime.interpreter as tflite
    full_tf = False

try:
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
except (ImportError, ModuleNotFoundError):
    from keras_preprocessing.image import load_img, img_to_array

import sys
import time
import numpy as np

interpreter_cls = tf.lite.Interpreter if full_tf else tflite.Interpreter
interpreter = interpreter_cls(model_path=sys.argv[1])

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
img = load_img(sys.argv[2], target_size=(72, 72))
x = img_to_array(img)
input_data = np.expand_dims(x, axis=0)
interpreter.set_tensor(input_details[0]['index'], input_data)

start_time = time.time()
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
end_time = time.time()

labels = ['LC', 'NLC']
rounded = int(round(prediction))
result = labels[rounded]

old_range = 0.5
new_range = 100.0

if rounded:
    pct = ((prediction - 0.5) * 100.0) / 0.5
else:
    pct = 100.0 - (prediction * 100.0 / 0.5)

print(f'elapsed time: {end_time - start_time:.2f}s, prediction: {result}, certainty: {pct:.2f}%')
