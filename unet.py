import os
import PIL
# checking for xrays and their respective masks
from glob import glob
import gradio as gr
import re
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import tensorflow as tf
from skimage import measure
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.activations import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

DIR = "./dataset"

lung_image_paths = glob(os.path.join(DIR, "Lung Segmentation/CXR_png/*.png"))
mask_image_paths = glob(os.path.join(DIR, "Lung Segmentation/masks/*.png"))

related_paths = defaultdict(list)

for img_path in lung_image_paths:
    img_match = re.search("CXR_png/(.*)\.png$", img_path)
    if img_match:
        img_name = img_match.group(1)
    for mask_path in mask_image_paths:
        mask_match = re.search(img_name, mask_path)
        if mask_match:
            related_paths["image_path"].append(img_path)
            related_paths["mask_path"].append(mask_path)

paths_df = pd.DataFrame.from_dict(related_paths)

# Preparing the training dataset
def prepare_train_test(df=pd.DataFrame(), resize_shape=tuple(), color_mode="rgb"):
    img_array = list()
    mask_array = list()

    for image_path in tqdm(paths_df.image_path):
        resized_image = cv2.resize(cv2.imread(image_path), resize_shape)
        resized_image = resized_image / 255.
        if color_mode == "gray":
            img_array.append(resized_image[:, :, 0])
        elif color_mode == "rgb":
            img_array.append(resized_image[:, :, :])

    for mask_path in tqdm(paths_df.mask_path):
        resized_mask = cv2.resize(cv2.imread(mask_path), resize_shape)
        resized_mask = resized_mask / 255.
        mask_array.append(resized_mask[:, :, 0])

    return img_array, mask_array


img_array, mask_array = prepare_train_test(df=paths_df, resize_shape=(256, 256), color_mode="gray")

# More on image preparation
img_train, img_test, mask_train, mask_test = train_test_split(img_array, mask_array, test_size=0.2, random_state=42)

img_side_size = 256
img_train = np.array(img_train).reshape(len(img_train), img_side_size, img_side_size)
img_test = np.array(img_test).reshape(len(img_test), img_side_size, img_side_size)
mask_train = np.array(mask_train).reshape(len(mask_train), img_side_size, img_side_size)
mask_test = np.array(mask_test).reshape(len(mask_test), img_side_size, img_side_size)

print(img_test[3])

### U-net model
def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

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


# # Model summary
EPOCHS = 3
model = unet(input_size=(256, 256, 1))
model.compile(optimizer=Adam(lr=5 * 1e-4), loss="binary_crossentropy", metrics=[dice_coef, 'binary_accuracy'])
model.summary()
#
# # Checkpoint and other functions
# tf.keras.utils.plot_model(model, to_file='model.png')

weight_path = "{}_weights.best.hdf5".format('cxr_reg')

checkpoint = ModelCheckpoint(weight_path, monitor='loss',  # verbose=1,
                             save_best_only=True,  # mode='min',
                             save_weights_only=True)

early = EarlyStopping(monitor="loss",
                      # mode="min",
                      patience=10)  # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early]
#
# # Model Training
history = model.fit(x=img_train,
                    y=mask_train,
                    validation_data=(img_test, mask_test),
                    epochs=EPOCHS,
                    batch_size=16,
                    callbacks=callbacks_list)
model.save('my_model.h5')


# Testing of model
def test_on_image(model, img_array, img_num, img_side_size=256):
    pred = model.predict(img_array[img_num].reshape(1, img_side_size, img_side_size, 1))
    pred[pred > 0.5] = 1.0
    pred[pred < 0.5] = 0.0
    fig = plt.figure(figsize=(15, 10))

    plt.subplot(1, 4, 1)
    plt.imshow(pred.reshape(img_side_size, img_side_size), cmap="Blues")
    plt.title("Previsão")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(mask_test[img_num].reshape(img_side_size, img_side_size), cmap="Blues")
    plt.title("Máscara real");
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(mask_test[img_num].reshape(img_side_size, img_side_size), cmap="Blues", alpha=0.5)
    plt.imshow(pred.reshape(img_side_size, img_side_size), cmap="PuBu", alpha=0.3)
    plt.title("Sobreposição")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(img_array[img_num].reshape(img_side_size, img_side_size), cmap="gray")
    plt.title("Original")
    plt.axis("off")
    plt.show()

    return pred


def dice_coef_test(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union == 0: return 1
    intersection = np.sum(y_true_f * y_pred_f)
    return 2. * intersection / union


IMG_NUM = 3  # Melhor img_num 12 (0.98) Pior img_num 10 (0.9)

prediction = test_on_image(model, img_array=img_test, img_num=IMG_NUM, img_side_size=256)
print(dice_coef_test(y_true=mask_test[IMG_NUM], y_pred=prediction))


def get_metrics(history):
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.legend()
    plt.xlabel("Época")
    plt.ylabel("Entropia cruzada binária")

    plt.subplot(2, 2, 2)
    plt.plot(history.history["dice_coef"], label="training dice coefficient")
    plt.plot(history.history["val_dice_coef"], label="validation dice coefficient")
    plt.legend()
    plt.xlabel("Época")
    plt.ylabel("Coeficiente Dice")
    plt.show()


get_metrics(history=history)
image = gr.inputs.Image(shape=(200,200))
label = gr.outputs.Label(num_top_classes=2)

gr.Interface(fn=predict_image, inputs=image, outputs=label,interpretation='default').launch(debug='True')