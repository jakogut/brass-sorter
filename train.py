from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, Activation
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_size = 72, 72
input_shape = *input_size, 3
num_classes = 1

def create_small():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

model = create_small()
model.summary()

batch_size = 256
epochs = 100

train_datagen = ImageDataGenerator(
    rescale=1.0,
    rotation_range=180,
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(rescale=0.25)

train_generator = test_datagen.flow_from_directory(
    'data/train', target_size=input_size, batch_size=32, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    'data/validation', target_size=input_size, batch_size=32, class_mode='binary') 

early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10,
        verbose=1, mode='auto')

checkpointer = ModelCheckpoint(filepath='brass_classifier.hdf5', verbose=1, save_best_only=True)
callbacks = [early_stop, checkpointer]

model.fit_generator(
    train_generator,
    steps_per_epoch=250,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=800,
    callbacks=callbacks
)
