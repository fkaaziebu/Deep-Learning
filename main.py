# necessary libraries
import pathlib
import gradio as gr
import tensorflow as tf
import numpy as np

from models.inceptionv3 import Inception
from models.chexnet import CheXNetModel
from models.densenet201 import DenseNet201Model

from data.data import Generate
from data.generate_unet_data import Generate as Gen
from utils.utils import Utils

base_path = "./dataset/TB_Chest_Radiography_Database"
DIR = "./dataset"
base_path = pathlib.Path(base_path)

# Generating the dataset
data = Generate(base_path=base_path, target_size=(224, 224), batch_size=16)
data1 = Gen(src=DIR)
li, train_data, val_data = data.generate_data()


# Instance of the models
inception = Inception(train_data=train_data, val_data=val_data, num_epochs=15)
densenet121 = CheXNetModel(train_data=train_data, val_data=val_data, num_epochs=15)
densenet201 = DenseNet201Model(train_data=train_data, val_data=val_data, num_epochs=15)

is_train = False
if is_train:
    # Training the model
    history = densenet201.forward()
    print(history.history.keys())

    # Statistics about the model, (plots, etc)
    utilities = Utils(hist=history, model=densenet201, validation_data=val_data, li=li)
    utilities.forward()
else:
    # Load model weight
    inception.load_weights()
    densenet121.load_weights()

    image = gr.inputs.Image(shape=(75, 75))

    # Gradio output labels
    # label = gr.outputs.Label(num_top_classes=2, label="Densenet201")
    label2 = gr.outputs.Label(num_top_classes=2, label="InceptionV3")
    label3 = gr.outputs.Label(num_top_classes=2, label="Densenet121")


    def predict_image(input_img):
        class_names = ['Normal', 'Tuberculosis']
        # print("The image to interpret: ", input_img)

        # Segmentation

        input_img = input_img.reshape(75, 75, -1)
        input_img = tf.keras.utils.img_to_array(input_img)
        input_img = np.expand_dims(input_img, axis=0)
        input_img = input_img / 255
        # prediction1 = densenet201.model.predict(input_img)
        prediction2 = inception.model.predict(input_img)
        prediction3 = densenet121.model.predict(input_img)
        # m1 = prediction1.flatten()
        m2 = prediction2.flatten()
        m3 = prediction3.flatten()

        # Densenet201
        # if m1 < 0.5:
        #     d = 1 - prediction1[0]
        #     prediction1 = np.insert(prediction1, 0, d)
        # else:
        #     d = 1 - prediction1[0]
        #     prediction1 = np.insert(prediction1, 0, d)

        # InceptionV3
        if m2 < 0.5:
            d = 1 - prediction2[0]
            prediction2 = np.insert(prediction2, 0, d)
        else:
            d = 1 - prediction2[0]
            prediction2 = np.insert(prediction2, 0, d)

        # # Densenet121
        if m3 < 0.5:
            d = 1 - prediction3[0]
            prediction3 = np.insert(prediction3, 0, d)
        else:
            d = 1 - prediction3[0]
            prediction3 = np.insert(prediction3, 0, d)

        # Return the predictions as a JSON string
        # print("Prediction: {")
        # print("Densenet201:", {class_names[i]: float(prediction1[i]) for i in range(2)})
        # print(",")
        # print("InceptionV3:", {class_names[i]: float(prediction2[i]) for i in range(2)})
        # print(",")
        # print("Chexnet:", {class_names[i]: float(prediction3[i]) for i in range(2)})
        # print("}")

        return {class_names[i]: float(prediction2[i]) for i in range(2)}, \
            {class_names[i]: float(prediction3[i]) for i in range(2)}

    gr.Interface(
        fn=predict_image,
        inputs=image,
        outputs=[label2, label3],
        interpretation='default'
    ).launch(debug='True')
