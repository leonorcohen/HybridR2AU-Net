from evaluation import *
from hybrid import *
from processing import * 

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add, ELU, LeakyReLU
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

from PIL import Image
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
###################################################
################################ LOADING THE IMAGES
###################################################

# Get the list of image and mask files - size 640*2560
img_files = next(os.walk('/scratch/mcohenmo/Hybrid/data/640x2560/original'))[2]
msk_files = next(os.walk('/scratch/mcohenmo/Hybrid/data/640x2560/masks'))[2]


# Sort the file lists
img_files.sort()
msk_files.sort()

# Print the number of image and mask files
print('Number of image files:', len(img_files))
print('Number of mask files:', len(msk_files))


# Initialize lists to store images and masks
X = []
Y = []

# Iterate over each image file
for img_fl in tqdm(img_files):
    # Check if the file has the PNG extension
    if img_fl.split('.')[-1] == 'png':
        # Load the image
        img = cv2.imread('/scratch/mcohenmo/Hybrid/data/640x2560/original/{}'.format(img_fl), cv2.IMREAD_GRAYSCALE)        
        
        # Append the image to the X list
        X.append(img)
        
        # Load and resize the corresponding mask
        msk = cv2.imread('/scratch/mcohenmo/Hybrid/data/640x2560/masks/{}'.format(img_fl), cv2.IMREAD_GRAYSCALE)
        
        # Append the mask to the Y list
        Y.append(msk)

        
# Print the number of samples in X and Y
print ('LOADED IMAGES')
print('Number of samples in X:', len(X))
print('Number of samples in Y:', len(Y))


###################################################
################################## TRAIN-TEST SPLIT
###################################################

# Convert the lists X and Y to NumPy arrays
X = np.array(X)
Y = np.array(Y)

print('CONVERTED NUMPY ARRAYS')
print(X.shape)
print(Y.shape)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=None)

# Reshape the Y arrays to add an extra dimension for channel
Y_train = Y_train.reshape((Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], 1))
Y_test = Y_test.reshape((Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], 1))

# Expand dimensions of X arrays to have shape (None, 2560, 640, 1)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Normalize the pixel values to the range [0, 1]
X_train = X_train / 255
X_test = X_test / 255
Y_train = Y_train / 255
Y_test = Y_test / 255

# Uncomment the following lines if you want to round the Y arrays to 0 or 1
#Y_train = np.round(Y_train, 0)
#Y_test = np.round(Y_test, 0)

# Print the shapes of the training and testing data
print('SHAPES BEFORE SLICING')
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

###################################################
########### SAVING TRAINING IMAGES IN NEW DIRECTORY
###################################################

output_dir_images = '/scratch/mcohenmo/Hybrid/data/processing/original'  # Specify the directory where you want to save the images from X_train
output_dir_masks = '/scratch/mcohenmo/Hybrid/data/processing/masks'  # Specify the directory where you want to save the images from Y_train

# Create the output directories if they don't exist
os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(output_dir_masks, exist_ok=True)

# Save the images from X_train
for i, image in enumerate(X_train):
    filename = f'image_{i}.png'
    output_file = os.path.join(output_dir_images, filename)
    cv2.imwrite(output_file, (image * 255).astype(np.uint8))

# Save the images from Y_train
for i, image in enumerate(Y_train):
    filename = f'mask_{i}.png'
    output_file = os.path.join(output_dir_masks, filename)
    cv2.imwrite(output_file, (image * 255).astype(np.uint8))


###################################################
########################### SLICING TRAINING IMAGES
###################################################   
    
input_img_path = "/scratch/mcohenmo/Hybrid/data/processing/original/"
input_mask_path = "/scratch/mcohenmo/Hybrid/data/processing/masks/"
output_img_path = "/scratch/mcohenmo/Hybrid/data/processing/original_sliced/"
output_mask_path = "/scratch/mcohenmo/Hybrid/data/processing/msk_sliced/"

# Create the output directories if they don't exist
os.makedirs(output_img_path, exist_ok=True)
os.makedirs(output_mask_path, exist_ok=True)

number_of_slices = 8

# Get the list of image files in the input directory
img_files = os.listdir(input_img_path)

# Call the slicing function
slicing(img_files, input_img_path, input_mask_path, output_img_path, output_mask_path, number_of_slices)

###################################################
########################### DELETING BLANK IMAGES
###################################################  
import shutil

# Path for saving deleted images and masks
deleted_img_path = "/scratch/mcohenmo/Hybrid/data/processing/deleted_images/"
deleted_mask_path = "/scratch/mcohenmo/Hybrid/data/processing/deleted_masks/"

# Create the directories for deleted images and masks if they don't exist
os.makedirs(deleted_img_path, exist_ok=True)
os.makedirs(deleted_mask_path, exist_ok=True)

# Function to check if a mask has all pixels equal to 0
def has_all_zeros(mask):
    return np.all(mask == 0)

# Threshold value for sum of pixel intensities - images below are not used for training 
threshold = 2000000

# Get the list of sliced image files in the output directory
sliced_img_files = os.listdir(output_img_path)

# Iterate over the sliced image files
for img_file in sliced_img_files:
    img_path = os.path.join(output_img_path, img_file)
    
    # Construct the corresponding mask file path based on the image file name
    mask_file = img_file.replace(".png", ".png")  # Modify this line if the mask file extension is different
    mask_path = os.path.join(output_mask_path, mask_file)
    '''
    Delete images with mask all zeros
    if os.path.isfile(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if has_all_zeros(mask):
            # Move the image and mask to the deleted directories
            shutil.move(img_path, os.path.join(deleted_img_path, img_file))
            shutil.move(mask_path, os.path.join(deleted_mask_path, mask_file))
    '''
    

    if os.path.isfile(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if has_sum_below_threshold(mask, threshold):
            # Move the image and mask to the deleted directories
            shutil.move(img_path, os.path.join(deleted_img_path, img_file))
            shutil.move(mask_path, os.path.join(deleted_mask_path, mask_file))


###################################################
################ LOADING THE SLICED TRAINING IMAGES
###################################################  

img_files = next(os.walk('/scratch/mcohenmo/Hybrid/data/processing/original_sliced'))[2]
msk_files = next(os.walk('/scratch/mcohenmo/Hybrid/data/processing/msk_sliced'))[2]

# Sort the file lists
img_files.sort()
msk_files.sort()

# Print the number of image and mask files
print('Number of image files:', len(img_files))
print('Number of mask files:', len(msk_files))

# Initialize lists to store images and masks
X = []
Y = []

# Iterate over each image file
for img_fl in tqdm(img_files):
    # Check if the file has the PNG extension
    if img_fl.split('.')[-1] == 'png':
        # Load the image
        img = cv2.imread('/scratch/mcohenmo/Hybrid/data/processing/original_sliced/{}'.format(img_fl), cv2.IMREAD_GRAYSCALE)        
        
        # Append the image to the X list
        X.append(img)
        
        # Load and resize the corresponding mask
        msk = cv2.imread('/scratch/mcohenmo/Hybrid/data/processing/msk_sliced/{}'.format(img_fl), cv2.IMREAD_GRAYSCALE)
                
        # Append the mask to the Y list
        Y.append(msk)

        
# Print the number of samples in X and Y
print('SHAPES OF TRAINING IMAGES AFTER SLICING')
print('Number of samples in X:', len(X))
print('Number of samples in Y:', len(Y))

# Convert the lists X and Y to NumPy arrays
X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)

# Reshape the Y arrays to add an extra dimension for channel
Y = Y.reshape((Y.shape[0], Y.shape[1], Y.shape[2], 1))

# Expand dimensions of X arrays to have shape (None, 2560, 640, 1)
X = np.expand_dims(X, axis=-1)

# Normalize the pixel values to the range [0, 1]
X = X / 255
Y = Y / 255

# Uncomment the following lines if you want to round the Y array to 0 or 1
# Y = np.round(Y, 0)

# Print the shapes of the training data
print('X_train shape:', X.shape)
print('Y_train shape:', Y.shape)

# Assign X_train and Y_train as the final outputs
X_train = X
Y_train = Y


# Print the shapes of the training and testing data
print('SHAPES AFTER SLICING')
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

#########################################################################
################ TRAIN MODEL WITH SLICED IMAGES AND TEST WITH FULL IMAGES
#########################################################################

# TRAINING
train_input_shape = (320, 640, 1)
train_model = hybrid(input_size = train_input_shape)

# TESTING 
test_input_shape = (2560, 640, 1)
test_model = hybrid(input_size = test_input_shape)

# Open the log.txt file and clear its contents
with open('models/log.txt', 'w') as fp:
    fp.close()
    
# Open the dice.txt file and clear its contents
with open('models/dice.txt', 'w') as fp:
    fp.close()

# Open the best.txt file, write '-1.0' as the initial best Jaccard Index value, and close the file
with open('models/best.txt', 'w') as fp:
    fp.write('-1.0')
    fp.close()

# Delete training.log if it exists
if os.path.exists('/scratch/mcohenmo/Hybrid/training.log'):
  os.remove('/scratch/mcohenmo/Hybrid/training.log')
  print("File Deleted!")
else:
  print("The file does not exist")


model = train_step(train_model,test_model, X_train, Y_train, X_test, Y_test, epochs=40, batch_size=2)

input("Press Enter to continue to last testing!")

# Save the weights of the best train_model
last_test_model = hybrid(input_size = test_input_shape)

# Load the saved weights into the last_test_model
last_test_model.load_weights('models/model_weights.h5')

# Testing with the model that achieved better results
evaluate_model(last_test_model, X_test, Y_test, batch_size=1)
