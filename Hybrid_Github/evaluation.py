import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred):
    smooth = 0.0  # Smoothing factor to prevent division by zero
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)  # Compute the intersection between y_true and y_pred
    dice = (2. * intersection + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)  # Compute the Dice coefficient
    return dice

def jacard(y_true, y_pred):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)  # Compute the intersection between y_true and y_pred
    union = K.sum(y_true_flat + y_pred_flat - y_true_flat * y_pred_flat)  # Compute the union of y_true and y_pred
    jaccard = intersection / union  # Compute the Jaccard index
    return jaccard

def save_model(model):
    # Convert the model architecture to JSON format
    model_json = model.to_json()

    # Create the 'models' directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save the model architecture to a JSON file
    with open('models/model_architecture.json', 'w') as fp:
        fp.write(model_json)

    # Save the model weights to an HDF5 file
    model.save_weights('models/model_weights.h5')
    

def evaluate_model(model, X_test, Y_test, batch_size):
    # Create a directory for storing the results if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Perform predictions using the model on the test data
    predictions = model.predict(x=X_test, batch_size=batch_size, verbose=1)

    # Round the predicted values to either 0 or 1 (black or white)
    predictions = np.round(predictions)

    # Iterate over each test sample
    for i in range(len(X_test)):
        
        input_image = X_test[i]
        ground_truth_mask = Y_test[i].reshape(Y_test[i].shape[0], Y_test[i].shape[1])
        predicted_mask = predictions[i].reshape(predictions[i].shape[0], predictions[i].shape[1])

        # Create a figure to display the input, ground truth, and prediction
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))

        # Plot the input image
        axes[0].imshow(input_image, cmap='gray')
        axes[0].set_title('Input')

        # Plot the ground truth mask
        axes[1].imshow(ground_truth_mask, cmap='binary')
        axes[1].set_title('Ground Truth')

        # Plot the predicted mask
        axes[2].imshow(predicted_mask, cmap='binary')
        axes[2].set_title('Prediction')

        # Calculate the Jaccard Index (Intersection over Union) for the prediction
        intersection = np.sum(predicted_mask * ground_truth_mask)
        #print("Intersection: ", intersection)
        union = np.sum(predicted_mask) + np.sum(ground_truth_mask) - intersection
        #print("Union: ", union)

        if union == 0 or union == 1:
            jaccard_index = 1
        else:
            jaccard_index = intersection / union

        
        # Calculate the Dice Coefficient for the prediction
        if union == 0: 
            dice_coefficient = 1
            print("Image with union = 0: ", i)
        elif union == 1:
            dice_coefficient = 1
            print("Image with union = 1: ", i)
        else:
            dice_coefficient = (2. * intersection) / (np.sum(predicted_mask) + np.sum(ground_truth_mask))


        # Set the title of the figure to include the Jaccard Index and Dice Coefficient
        title = f"Jaccard Index: {intersection}/{union} = {jaccard_index:.4f}\nDice Coefficient: {dice_coefficient:.4f}"
        fig.suptitle(title)

        # Save the figure as a PNG file
        plt.savefig(f'results/{i}.png')
        # Close the figure to free up resources
        plt.close(fig)

        # Create a figure just for the prediction
        fig, axes = plt.subplots(1, 1, figsize=(4, 8))

        # Plot the input image
        axes.imshow(predicted_mask, cmap='binary')
        #axes.set_title('Attention U-Net')

        # Save the figure as a PNG file
        plt.savefig(f'results/pred/{i}.png')

        # Close the figure to free up resources
        plt.close(fig)


    # Calculate the average Jaccard Index and Dice Coefficient for all test samples
    jaccard_avg = 0
    dice_avg = 0
    for i in range(len(Y_test)):
        predicted_mask = predictions[i].ravel()
        ground_truth_mask = Y_test[i].ravel()

        intersection = predicted_mask * ground_truth_mask
        union = predicted_mask + ground_truth_mask - intersection

        if np.sum(union) == 0 or np.sum(union) == 1:
            jaccard_index = 1  # or any other desired value
            dice_coefficient = 1  # or any other desired value
        else:
            jaccard_index = np.sum(intersection) / np.sum(union)
            dice_coefficient = (2. * np.sum(intersection)) / (np.sum(predicted_mask) + np.sum(ground_truth_mask))

        jaccard_avg += jaccard_index
        dice_avg += dice_coefficient

    jaccard_avg /= len(Y_test)
    dice_avg /= len(Y_test)

    # Print the average Jaccard Index and Dice Coefficient
    print('Jaccard Index: ', jaccard_avg)
    print('Dice Coefficient: ', dice_avg)

    # Append the Jaccard Index to the log file
    with open('models/log.txt', 'a') as fp:
        fp.write(str(jaccard_avg) + '\n')
        
    # Append the Dice score to the file
    with open('models/dice.txt', 'a') as fp:
        fp.write(str(dice_avg) + '\n')

    # Read the best Jaccard Index from the best.txt file
    with open('models/best.txt', 'r') as fp:
        best_jaccard = float(fp.read())

    # Check if the current Jaccard Index is better than the best one so far
    if jaccard_avg > best_jaccard:
        print('***********************************************')
        print('Jaccard Index improved from', best_jaccard, 'to', jaccard_avg)
        print('***********************************************')

        # Write the new best Jaccard Index to the best.txt file
        with open('models/best.txt', 'w') as fp:
            fp.write(str(jaccard_avg))

        # Save the current model
        save_model(model)
        
csv_logger = CSVLogger('training.log', append=True)

def train_step(train_model, test_model, X_train, Y_train, X_test, Y_test, epochs, batch_size):
    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}')
        # Training
        train_model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=1, validation_split=0.1, verbose=1, callbacks = [csv_logger])
        
        # Save the weights of the train_model
        train_model.save_weights('train_model_weights.h5')
        
        # Load the saved weights into the test_model
        test_model.load_weights('train_model_weights.h5')

        # Testing
        evaluate_model(test_model, X_test, Y_test, batch_size=1)
    return train_model
        
