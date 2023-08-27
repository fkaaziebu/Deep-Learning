import splitfolders

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from models.auriscope import Auriscope
import shutil
import gradio as gr
import tensorflow as tf

splitfolders.ratio(
    "./data-phil/Datos/Training-validation",
    output="MultiDisease",
    seed=1337,
    ratio=(.8, .1, .1), #80% for training, 10% for validation and 10% for testing.
    group_prefix=None
)

channels = 3
batch_size = 25
test_batch_size = 32
test_steps = 1
train_path = './MultiDisease/train'
test_path = './MultiDisease/test'
val_path = './MultiDisease/val'


def scalar(img): #Generating images for preprocessing for the model
    return img


trgen = ImageDataGenerator(
    preprocessing_function=scalar,
    horizontal_flip=True
)

tvgen = ImageDataGenerator(preprocessing_function=scalar)

train_generator = trgen.flow_from_directory(
    directory=train_path,
    target_size=(224, 224),
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    batch_size=batch_size
)

test_generator = tvgen.flow_from_directory(
    directory=test_path,
    target_size=(224, 224),
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False,
    batch_size=test_batch_size
)

valid_generator = tvgen.flow_from_directory(
    directory=val_path,
    target_size=(224, 224),
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    batch_size=batch_size
)

classes = list(train_generator.class_indices.keys())
class_count = len(classes)
train_steps = int(np.ceil(len(train_generator.labels) / batch_size))

model = Auriscope(train_data=train_generator, val_data=valid_generator)

def print_info(test_gen, preds, print_code, save_dir, subject):
    class_dict = test_gen.class_indices
    labels = test_gen.labels
    file_names = test_gen.filenames
    error_list = []
    true_class = []
    pred_class = []
    prob_list = []
    new_dict = {}
    error_indices = []
    y_pred = []
    for key, value in class_dict.items():
        new_dict[value] = key  # dictionary {integer of class number: string of class name}
    # store new_dict as a text fine in the save_dir
    classes = list(new_dict.values())  # list of string of class names
    errors = 0
    for i, p in enumerate(preds):
        pred_index = np.argmax(p)
        true_index = labels[i]  # labels are integer values
        if pred_index != true_index:  # a misclassification has occurred
            error_list.append(file_names[i])
            true_class.append(new_dict[true_index])
            pred_class.append(new_dict[pred_index])
            prob_list.append(p[pred_index])
            error_indices.append(true_index)
            errors = errors + 1
        y_pred.append(pred_index)
    if print_code != 0:
        if errors > 0:
            if print_code > errors:
                r = errors
            else:
                r = print_code
            msg = '{0:^28s}{1:^28s}{2:^28s}{3:^16s}'.format('Filename', 'Predicted Class', 'True Class', 'Probability')
            print_in_color(msg, (0, 255, 0), (55, 65, 80))
            for i in range(r):
                split1 = os.path.split(error_list[i])
                split2 = os.path.split(split1[0])
                fname = split2[1] + '/' + split1[1]
                msg = '{0:^28s}{1:^28s}{2:^28s}{3:4s}{4:^6.4f}'.format(fname, pred_class[i], true_class[i], ' ',
                                                                       prob_list[i])
                print_in_color(msg, (255, 255, 255), (55, 65, 60))
                # print(error_list[i]  , pred_class[i], true_class[i], prob_list[i])
        else:
            msg = 'With accuracy of 100 % there are no errors to print'
            print_in_color(msg, (0, 255, 0), (55, 65, 80))
    if errors > 0:
        plot_bar = []
        plot_class = []
        for key, value in new_dict.items():
            count = error_indices.count(key)
            if count != 0:
                plot_bar.append(count)  # list containing how many times a class c had an error
                plot_class.append(value)  # stores the class
        fig = plt.figure()
        fig.set_figheight(len(plot_class) / 3)
        fig.set_figwidth(10)
        plt.style.use('fivethirtyeight')
        for i in range(0, len(plot_class)):
            c = plot_class[i]
            x = plot_bar[i]
            plt.barh(c, x, )
            plt.title(' Errors by Class on Test Set')
    y_true = np.array(labels)
    y_pred = np.array(y_pred)
    if len(classes) <= 30:
        # create a confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        length = len(classes)
        if length < 8:
            fig_width = 8
            fig_height = 8
        else:
            fig_width = int(length * .5)
            fig_height = int(length * .5)
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(length) + .5, classes, rotation=90)
        plt.yticks(np.arange(length) + .5, classes, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    clr = classification_report(y_true, y_pred, target_names=classes)
    print("Classification Report:\n----------------------\n", clr)


def print_in_color(txt_msg, fore_tupple, back_tupple, ):
    # prints the text_msg in the foreground color specified by fore_tupple with the background specified by back_tupple
    # text_msg is the text, fore_tupple is foregroud color tupple (r,g,b), back_tupple is background tupple (r,g,b)
    rf, gf, bf = fore_tupple
    rb, gb, bb = back_tupple
    msg = '{0}' + txt_msg
    mat = '\33[38;2;' + str(rf) + ';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' + str(gb) + ';' + str(
        bb) + 'm'
    print(msg.format(mat), flush=True)
    print('\33[0m', flush=True)  # returns default print color to back to black
    return

def saver(save_path, model, model_name, subject, accuracy, img_size, scalar, generator):
    # first save the model
    save_id = str(model_name + '-' + subject + '-' + str(acc)[:str(acc).rfind('.') + 3] + '.h5')
    model_save_loc = os.path.join(save_path, save_id)
    model.save(model_save_loc)
    print_in_color('model was saved as ' + model_save_loc, (0, 255, 0), (55, 65, 80))
    # now create the class_df and convert to csv file
    class_dict = generator.class_indices
    height = []
    width = []
    scale = []
    for i in range(len(class_dict)):
        height.append(img_size[0])
        width.append(img_size[1])
        scale.append(scalar)
    Index_series = pd.Series(list(class_dict.values()), name='class_index')
    Class_series = pd.Series(list(class_dict.keys()), name='class')
    Height_series = pd.Series(height, name='height')
    Width_series = pd.Series(width, name='width')
    Scale_series = pd.Series(scale, name='scale by')
    class_df = pd.concat([Index_series, Class_series, Height_series, Width_series, Scale_series], axis=1)
    csv_name = 'class_dict.csv'
    csv_save_loc = os.path.join(save_path, csv_name)
    class_df.to_csv(csv_save_loc, index=False)
    print_in_color('class csv file was saved as ' + csv_save_loc, (0, 255, 0), (55, 65, 80))
    return model_save_loc, csv_save_loc


def tr_plot(tr_data, start_epoch):
    # Plot the training and validation data
    tacc = tr_data.history['accuracy']
    tloss = tr_data.history['loss']
    vacc = tr_data.history['val_accuracy']
    vloss = tr_data.history['val_loss']
    Epoch_count = len(tacc) + start_epoch
    Epochs = []
    for i in range(start_epoch, Epoch_count):
        Epochs.append(i + 1)
    index_loss = np.argmin(vloss)  # this is the epoch with the lowest validation loss
    val_lowest = vloss[index_loss]
    index_acc = np.argmax(vacc)
    acc_highest = vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
    vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    axes[0].plot(Epochs, tloss, 'r', label='Training loss')
    axes[0].plot(Epochs, vloss, 'g', label='Validation loss')
    axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(Epochs, tacc, 'r', label='Training Accuracy')
    axes[1].plot(Epochs, vacc, 'g', label='Validation Accuracy')
    axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s=150, c='blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout
    # plt.style.use('fivethirtyeight')
    plt.show()

is_train = False
#Where we either train or test our model
if is_train:
    history = model.forward()
    sns.set_style('darkgrid')
    p = model.predict(test_generator)
    print_info(test_generator, p, 10, r'./', 'MultiDisease')

    model.evaluate(test_generator)

    tr_plot(history, 0)
    subject = 'ear disease'
    working_dir = r'./'
    img_size = (224, 224)
    model_name = "Inception Model"
    acc = model.evaluate(test_generator, verbose=1, steps=test_steps, return_dict=False)[1] * 100
    msg = f'accuracy on the test set is {acc:5.2f} %'
    print_in_color(msg, (0, 255, 0), (55, 65, 80))
    generator = train_generator
    scale = 1
    model_save_loc, csv_save_loc = saver(working_dir, model, model_name, subject, acc, img_size, scale, generator)
    shutil.rmtree('./MultiDisease/')
else:
    image = gr.inputs.Image(shape=(75, 75))
    label = gr.outputs.Label(num_top_classes=4, label="InceptionV3")
# This function is for the user interface Gradio
    def predict_image(input_img):
        class_names = ['Chronic otitis media', 'Earwax plug', 'Myringosclerosis', 'Normal']
        input_img = input_img.reshape(75, 75, -1)
        input_img = tf.keras.utils.img_to_array(input_img)
        input_img = np.expand_dims(input_img, axis=0)
        input_img = input_img / 255
        prediction = model.model.predict(input_img)
        m = prediction.flatten()[0]

        if m < 0.5:
            d = 1 - prediction[0][0][0][0]
            prediction = np.insert(prediction, 0, d)
        else:
            d = 1 - prediction[0][0][0][0]
            prediction = np.insert(prediction, 0, d)

        return {class_names[i]: float(prediction[i]) for i in range(4)}
    gr.Interface(
        fn=predict_image,
        inputs=image,
        outputs=label,
        interpretation="default"
    ).launch(debug="True")

