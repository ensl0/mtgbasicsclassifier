from tensorflow import keras
import numpy as np
import os
from PIL import Image
import requests


# This class holds the model
class Basics_Model:
    data_train = None
    data_val = None
    image_size = (100, 100)
    batch_size = 32

    def __init__(self):
        self.data_train = keras.preprocessing.image_dataset_from_directory(
            'S:\MTGARTLANDS',
            labels="inferred",
            label_mode='categorical',
            validation_split=0.2,
            subset="training",
            seed=100,
            image_size=self.image_size,
            batch_size=self.batch_size,
        )

        self.data_val = keras.preprocessing.image_dataset_from_directory(
            'S:\MTGARTLANDS',
            labels="inferred",
            label_mode='categorical',
            validation_split=0.2,
            subset="validation",
            seed=100,
            image_size=self.image_size,
            batch_size=self.batch_size,
        )

    # This model takes the rgb channels of a processed image as input. Out tries to categorize the card into colors

    def Init_Model(self):
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)

        data_augmentation = keras.Sequential(
            [
                keras.layers.RandomFlip(),
                keras.layers.RandomRotation(0.3),
            ]
        )

        augmented_train_ds = self.data_train.map(
            lambda x, y: (data_augmentation(x, training=True), y))

        train_ds = augmented_train_ds.prefetch(buffer_size=32)
        val_ds = self.data_val.prefetch(buffer_size=32)

        model = keras.Sequential(
            [
                keras.Input(shape=self.image_size + (3,)),
                keras.layers.Rescaling(1. / 255),

                keras.layers.Dense(27, bias_initializer='ones'),

                keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
                keras.layers.MaxPooling2D(3),
                keras.layers.BatchNormalization(),

                keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
                keras.layers.BatchNormalization(),

                keras.layers.MaxPooling2D(3, padding="same"),

                keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
                keras.layers.MaxPooling2D(3),
                keras.layers.BatchNormalization(),

                keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
                keras.layers.BatchNormalization(),

                keras.layers.Flatten(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(5, activation="softmax")
            ]
        )

        model.summary()
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(train_ds, batch_size=32, callbacks=[callback], epochs=100, validation_data=val_ds,
                  validation_split=0.1)
        model.save(os.getcwd() + '//Models//')

    def test_Model(self):
        model = keras.models.load_model(os.getcwd() + '//Models//')
        model.evaluate(self.data_val)

    def predict_model(self, img_uri):
        d = {0: 'Forest', 1: 'Island', 2: 'Mountain', 3: 'Plains', 4: 'Swamp'}
        model = keras.models.load_model(os.getcwd() + '//Models//')
        r = requests.get(img_uri, stream=True)
        image = Image.open(r.raw)
        sized = image.resize(self.image_size)
        array = keras.preprocessing.image.img_to_array(sized)

        prediction = model.predict(array[None, :, :, :])
        print(prediction[0])
        ind = np.argmax(prediction[0])
        print(d[ind])
