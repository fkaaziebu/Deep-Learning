from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras import backend as keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np


class Unet:
    def __init__(self, input_size, img_train, mask_train, img_test, mask_test, batch_size, num_epochs):
        self.input_size = input_size
        self.model = Model()
        self.img_train = img_train
        self.mask_train = mask_train
        self.img_test = img_test
        self.mask_test = mask_test
        self.batch_size = batch_size
        self.epochs = num_epochs

    def create_unet_model(self):
        inputs = Input(self.input_size)

        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        return Model(inputs=[inputs], outputs=[conv10])

    def dice_coef(self, y_true, y_pred):
        y_true_f = keras.flatten(y_true)
        y_pred_f = keras.flatten(y_pred)
        intersection = keras.sum(y_true_f * y_pred_f)
        return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

    def segment_xray_image(self, img_array, img_num, img_side_size=256):
        pred = self.model.predict(img_array[img_num].reshape(1, img_side_size, img_side_size, 1))
        pred[pred > 0.5] = 1.0
        pred[pred < 0.5] = 0.0
        return pred.reshape(img_side_size, img_side_size)

    def load_weights(self):
        self.model = self.create_unet_model()
        self.model.compile(
            optimizer=Adam(lr=5 * 1e-4),
            loss="binary_crossentropy",
            metrics=[self.dice_coef, 'binary_accuracy']
        )

        weight_path = "{}_weights.best.hdf5".format('cxr_reg')
        self.model.load_weights(filepath=weight_path)

    def predict(self, input_img, img_side_size=256):
        pred = self.model.predict(input_img.reshape(1, img_side_size, img_side_size, 1))
        pred[pred > 0.5] = 1.0
        pred[pred < 0.5] = 0.0
        return pred.reshape(img_side_size, img_side_size)

    def forward(self):
        self.model = self.create_unet_model()
        self.model.compile(
            optimizer=Adam(lr=5 * 1e-4),
            loss="binary_crossentropy",
            metrics=[self.dice_coef, 'binary_accuracy']
        )

        weight_path = "{}_weights.best.hdf5".format('cxr_reg')

        checkpoint = ModelCheckpoint(
            weight_path,
            monitor='loss',
            save_best_only=True,
            save_weights_only=True
        )

        early = EarlyStopping(
            monitor="loss",
            patience=10
        )

        callbacks_list = [checkpoint, early]

        history = self.model.fit(
            x=self.img_train,
            y=self.mask_train,
            validation_data=(self.img_test, self.mask_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks_list
        )
        self.model.save('my_model.h5')

        return history
