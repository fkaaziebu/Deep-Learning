import tensorflow as tf
from keras import optimizers
from keras import Sequential
from keras.applications.xception import Xception
from keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
import numpy as np


class Auriscope:
    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data
        self.model = Sequential()

    def layers(self):
        base_model = Xception(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
            classes=4
        )
        self.model.add(base_model)
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation=('relu'), input_dim=512))
        self.model.add(Dense(512, activation=('relu')))
        self.model.add(Dense(256, activation=('relu')))
        self.model.add(Dropout(.3))
        self.model.add(Dense(128, activation=('relu')))
        self.model.add(Dense(4, activation=('softmax')))

    def load_weights(self):
        self.layers()
        sgd = SGD(learning_rate=.001, momentum=.9, nesterov=False)
        adam = tf.keras.optimizers.legacy.Adam(
            learning_rate=.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,
            amsgrad=False
        )
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        weight_path = 'inceptionv3.h5'
        self.model.load_weights(filepath=weight_path)
        self.predict()

    def predict(self):
        image_path = "./dataset/TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-100.png"
        new_img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        img = tf.keras.utils.img_to_array(new_img)
        img = np.expand_dims(img, axis=0)
        img = img / 255

        # print("Following is our prediction:")
        prediction = self.model.predict(img)
        return prediction, new_img

    def forward(self):
        sgd = SGD(learning_rate=.001, momentum=.9, nesterov=False)
        adam = tf.keras.optimizers.legacy.Adam(
            learning_rate=.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.0,
            amsgrad=False
        )
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.train_data, epochs=10, validation_data=self.val_data, verbose=1)
        self.model.fit(self.train_data, initial_epoch=10, epochs=15, validation_data=self.val_data, verbose=1)
        self.model.fit(self.train_data, initial_epoch=15, epochs=20, validation_data=self.val_data, verbose=1)
        history = self.model.fit(self.train_data, initial_epoch=20, epochs=25, validation_data=self.val_data, verbose=1)

        self.model.save("baseline_auriscope.h5")

        return history
