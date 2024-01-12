import os
import numpy as np
import random
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import joblib


# Noise Adding
def add_noise(images, mean=0, std=10):
    num_images, height, width = images.shape
    gauss = np.random.normal(mean, std, (num_images, height, width))
    images = np.clip(images + gauss, 0, 255).astype(np.uint8)
    return images

# Image Flipping
def flip(images):
    flip_type = random.choice(['horizontal', 'vertical'])
    if flip_type == 'horizontal':
        # Flip the image horizontally
        return images[:, :, ::-1]
    else:
        # Flip the image vertically
        return images[:, ::-1, :]

# Image Rotating
def rotate(images):
    # Corresponding to 90, 180, 270 degrees
    angle = random.choice([1, 2, 3])  # Corresponding to 90, 180, 270 degrees

    # Rotate the image by the selected angle
    images = np.rot90(images, angle)

    return images

# Preprocessing for SVM
def preprocessing_SVM(images, labels, train=True):
    num_data = len(images)
    num_augment = 3

    if train == True:
        images = np.repeat(images, num_augment, axis=0)
        labels = np.repeat(labels, num_augment, axis=0)

        images = add_noise(images)
        images[ 1 * num_data : 2 * num_data] = flip(images[ 1 * num_data : 2 * num_data])
        images[ 2 * num_data : 3 * num_data] = flip(images[ 2 * num_data : 3 * num_data])

    # Normalize the images to be in the range [0, 1]
    images = images.astype('float32') / 255.0
    images = images.reshape((images.shape[0], -1))
    labels = labels.ravel()

    return images, labels


# Train the SVM classifier and Evaluate the model
def A_SVM_training(train_images, train_labels, val_images, val_labels,
          C = 10,  # The regularization parameter.
          kernel = 'rbf',  # The type of kernel used.
          gamma = 'scale',  # Defines how far the influence of a single training example reaches.
          ):
    
    # Inputs Preprocessing
    train_images, train_labels = preprocessing_SVM(train_images, train_labels, train=True)
    val_images, val_labels = preprocessing_SVM(val_images, val_labels, train=False)

    # Initialize and train the SVM classifier
    model = svm.SVC(C=C, kernel=kernel, gamma=gamma)
    model.fit(train_images, train_labels)

    # Gets the root_path where the file is located
    file_path = os.path.abspath(__file__)
    root_path = os.path.dirname(file_path)

    # Saving the entire model 
    joblib.dump(model, os.path.join(root_path, "A_SVM_model.joblib"))

    # Evaluate the model
    val_predictions = model.predict(val_images)
    print(f"Accuracy Score: {accuracy_score(val_labels, val_predictions)}")
    print(f"Classification Report:\n{classification_report(val_labels, val_predictions)}")

    return model


# Test the SVM classifier
def A_SVM_testing(model, test_images, test_labels):
    # Inputs Preprocessing
    test_images, test_labels = preprocessing_SVM(test_images, test_labels, train=False)

    # Evaluate the model
    test_predictions = model.predict(test_images)
    print(f"Accuracy Score: {accuracy_score(test_labels, test_predictions)}")
    print(f"Classification Report:\n{classification_report(test_labels, test_predictions)}")

    return model
