import os
import cv2
import shutil
import math
import random
import pickle
import tensorflow as tf
import argparse
import keras
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from scipy.misc import imresize
from keras.models import model_from_json

# import our own modules
# import data
import face
# import cnn_model

# process images and split them into training, validation and test sets
# total_count = data.process_data()
# data.split_data(total_count)

# create the training and validation sets for race recognition
print("Begin creating training and validation sets")
file_path = "colour_files/model_sets.txt"
if os.path.exists(file_path): # if file exists, simply load file containing the sets 
	print("sets file already exists")
	# open file and load sets dictionary
	new_f = open(file_path, "rb")
	sets = pickle.load(new_f) 
	new_f.close()

	# initialize set variables
	train_x = sets["train_x"]
	train_y = sets["train_y"]
	valid_x = sets["valid_x"]
	valid_y = sets["valid_y"]
# else: # create the sets and store them in a pickle file for faster access 
# 	# initialize training set and validaiton set and also normalize their values 
# 	print("Sets file doesn't exist, now creating them")
# 	(train_x, train_y) = cnn_model.build_data("training", "all")
# 	(valid_x, valid_y) = cnn_model.build_data("validation", "all")
# 	train_x = train_x.reshape(train_x.shape[0], 50, 50, 3).astype('float32') / 255.0
# 	valid_x = valid_x.reshape(valid_x.shape[0], 50, 50, 3).astype('float32') / 255.0

# 	# store the sets in a pickle file for faster access 
# 	model_sets = {"train_x": train_x, "train_y": train_y, "valid_x": valid_x,
# 	           "valid_y": valid_y}
# 	new_f = open("{}".format(file_path), "wb") 
# 	pickle.dump(model_sets, new_f) 
# 	new_f.close()

# check if model exists
model_path = "colour_files/model.json"
if os.path.exists(model_path): 
	# if file exists, simply load file containing the model
	print("Race Recognition model already created")
	# load json and create model
	json_file = open(model_path, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights("colour_files/model.h5")
	print("Loaded race model from disk")
	 
	# evaluate loaded model on test data
	model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), 
	                    loss='sparse_categorical_crossentropy',
	                    metrics=['accuracy'])

# else:
# 	# initialize the model and train the network for weights and make predictions 
# 	# on the detected faces
# 	print("Initializing network")
# 	model = cnn_model.initialize_model()
# 	model = cnn_model.train_network(model, train_x, train_y, valid_x, valid_y, 30)

# 	# serialize model to JSON and weights to HDF5 for faster access
# 	model_json = model.to_json()
# 	with open(model_path, "w") as json_file:
# 	    json_file.write(model_json) 
# 	model.save_weights("colour_files/model.h5")
# 	print("Saved race model to disk")

# take in commandline input and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", required=True,
	help="path to the input image")
parser.add_argument("-p2", "--path2", required=True,
	help="path to the output image")
args = vars(parser.parse_args())

# initialize variables and detect the faces in the given image 
# input_img_path = "../test_photos/group_photo7.jpg"
input_img_path = args["path"]
detections = face.detect_faces(input_img_path)
test_x = np.zeros((len(detections), 50, 50, 3))
img = cv2.imread(input_img_path)
race_list = ["white", "black", "asian", "indian", "others"]

# loop over each detected face, perform face alignment and resize
for i, box in enumerate(detections):
	# find facial landmarks for detected face, and process the image
	landmarks = face.find_landmarks(input_img_path, box) 
	aligned_face = face.align_face(input_img_path, landmarks) # align face
	result = imresize(aligned_face, (50, 50, 3))

	# use the trained model to make a prediciton on the race of the detected face
	temp_face = np.array([list(result)])
	races = model.predict(temp_face)
	race_idx = np.argmax(races[0])

	# draw the bounding box around each face 
	(left, top, right, bottom) = box.astype("int")
	cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

	# show the race above every detected face
	cv2.putText(img, "{}".format(race_list[race_idx]), (left - 10, top - 10), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  
# cv2.imwrite('result.jpg', img)
cv2.imwrite(args["path2"], img)
print("image saved")