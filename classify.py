# necessary libraries
import pathlib
import gradio as gr
import tensorflow as tf
import numpy as np
import splitfolders

from models.image_classify import ImageClassifier
from data.data import Generate
from utils.utils import Utils

base_path = "./image_classify"
base_path = pathlib.Path(base_path)

splitfolders.ratio(
    base_path,
    output='X_ray_Cls',
    seed=123,
    ratio=(0.7, 0.15, 0.15),
    group_prefix=None
)

# Generating the dataset
data = Generate(base_path=base_path, target_size=(224, 224), batch_size=16)
li, train_data, val_data = data.generate_data()

# Instance of the inception model
classify_img = ImageClassifier(train_data=train_data, val_data=val_data, num_epochs=1, verbose=1)
is_train = False
if is_train:
    # Training the model
    history = classify_img.forward()
    print(history.history.keys())

    # Statistics about the model, (plots, etc)
    utilities = Utils(hist=history, model=classify_img, validation_data=val_data, li=li)
    utilities.forward()
else:
    # Load model weight
    image_path = "./image_classify/xray/CHNCXR_0001_0.png"
    new_img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img = tf.keras.utils.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    classify_img.load_weights()
    prediction = classify_img.model.predict(img)
    print(prediction[0][0])
