import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout

def imresize(image, size):
    pil_image = Image.fromarray(image)
    return np.array(pil_image.resize((size[1], size[0]), Image.ANTIALIAS))

def build_data(data_type, size):
    # initialize variables and file paths
    if data_type == "training":
        input_path = "../training_set/"
    elif data_type == "validation":
        input_path = "../validation_set/"
    elif data_type == "test":
        input_path = "../test_set/"
    races = ["white", "black", "asian", "indian", "others"]
    race_dict = {}
    labels_dict = {}
    race_size_dict = {}

    # loop over each race, and append the images
    for race in races:
        # initialize variables
        print("start making {} data for {}".format(data_type, race))
        race_path = "{}{}/".format(input_path, race)
        image_list = os.listdir(race_path)
        # determine the actual number of training examples in race
        if size == "all" and ".DS_Store" in image_list:
            race_size = len(image_list) - 1
        elif size == "all":  # .DS_Store not in image_list
            race_size = len(image_list)
        elif ".DS_Store" in image_list and size >= len(image_list):
            race_size = len(image_list) - 1
        else:
            race_size = min(size, len(image_list))
        temp = np.zeros((race_size, 50, 50, 3))
        temp_labels = np.zeros((1, race_size))
        race_size_dict[race] = race_size
        # loop over each image in the specific race folder
        for (i, img_name) in enumerate(image_list):
            # exits for loop when enough data is acquired
            if i == race_size:
                break

            # checks if file is actually an image
            if img_name != ".DS_Store":
                # extract the true label for the image
                labels = img_name.split("_")
                race_idx = int(labels[0])
                temp_labels[0, i] = race_idx

                # read and resize the image, and apply canny edge detection
                img_path = race_path + img_name
                img = cv2.imread(img_path)
                output = imresize(img, (50, 50, 3))
                # add img to temp
                temp[i] = output
                race_dict[race] = temp
        labels_dict[race] = temp_labels

    print("build data from all races")
    # create the actual matrix and vector that contain all race data and label
    actual_size = sum(race_size_dict.values())
    data = np.zeros((actual_size, 50, 50, 3))
    data_labels = np.zeros((1, actual_size))

    # append all race data into data
    end_idx, start_idx = 0, 0
    for race in races:
        end_idx += race_size_dict[race]
        start_idx = end_idx - race_size_dict[race]
        data[start_idx:end_idx] = race_dict[race]
        data_labels[0, start_idx:end_idx] = labels_dict[race]

    print("appended {} to data and labels".format(race))

    return (data, data_labels[0])

def initialize_model():
    model = tf.keras.models.Sequential([Conv2D(filters=64, kernel_size=3,
                                               activation=tf.nn.relu,
                                               padding='same',
                                               input_shape=(50, 50, 3)),
                                        Conv2D(filters=64, kernel_size=3,
                                               padding='same',
                                               activation=tf.nn.relu),
                                        MaxPooling2D(pool_size=2),
                                        Conv2D(filters=128, kernel_size=3,
                                               activation=tf.nn.relu,
                                               padding='same'),
                                        Conv2D(filters=128, kernel_size=3,
                                               activation=tf.nn.relu,
                                               padding='valid'),
                                        MaxPooling2D(pool_size=2),
                                        Conv2D(filters=32, kernel_size=3,
                                               activation=tf.nn.relu,
                                               padding='valid'),
                                        Dropout(0.2),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                        Dropout(0.2),
                                        tf.keras.layers.Dense(5, activation=tf.nn.softmax)])

    return model

def train_network(model, train_x, train_y, valid_x, valid_y, b_size):
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    fitted_model = model.fit(train_x, train_y, batch_size=b_size, epochs=15,
                             validation_data=(valid_x, valid_y), verbose=1)
    print("plotting graph now")
    # plot the training and validation Loss
    plt.plot(fitted_model.history['loss'])
    plt.plot(fitted_model.history['val_loss'])
    plt.title('Model Loss with batch size = {}'.format(b_size))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Test'])
    plt.show()
    # plot the training and validation accuracy
    plt.plot(fitted_model.history['accuracy'])
    plt.plot(fitted_model.history['val_accuracy'])
    plt.title('Model Accuracy with batch size = {}'.format(b_size))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'])
    plt.show()
    return model
