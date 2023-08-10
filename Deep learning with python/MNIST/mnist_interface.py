import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from keras import layers, models

network = models.load_model('mnist.h5')

image = gr.inputs.Image(shape=(28, 28))
label = gr.outputs.Label(num_top_classes=10, label='MNIST')


def predict_image(input_img):
    input_img = input_img.reshape(-1, 28 * 28)
    print(input_img.shape)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    prediction = network.predict(input_img)[0]
    return {class_names[i]: float(prediction[i]) for i in range(10)}


gr.Interface(fn=predict_image, inputs=image, outputs=[label], interpretation='default').launch(debug='True')
