'''
Loads a trained model, and classifies an image

argv[1]: path to hdf5 model to load
argv[2]: path to image to classify
'''

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

import sys
import time
import numpy as np

model = load_model(sys.argv[1])
img = load_img(sys.argv[2], target_size=(72, 72))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)

start_time = time.time()
prediction = model.predict(x, verbose=1)[0][0]
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
