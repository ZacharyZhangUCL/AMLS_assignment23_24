# Medical Image Analysis Project

## Project Description
This project focuses on the application of machine learning models for medical image analysis. It utilizes two datasets, PneumoniaMNIST and PathMNIST, to demonstrate the implementation of various models including Support Vector Machines (SVM), Convolutional Neural Networks (CNN), and Random Forests.

## File Structure
- `main.py`: The main executable script for the project. It handles data loading, model selection, training, and testing.
- `A/`:
  - `A_SVM.py`: Contains functions for training and testing the SVM model on the PneumoniaMNIST dataset.
  - `A_CNN.py`: Contains functions for training and testing the CNN model on the PneumoniaMNIST dataset.
- `B/`:
  - `B_RandomForest.py`: Contains functions for training and testing the Random Forest model on the PathMNIST dataset.
  - `B_CNN.py`: Contains functions for training and testing the CNN model on the PathMNIST dataset.
- `Datasets/`: Directory where the PneumoniaMNIST and PathMNIST datasets are stored.

## How to Run
1. Set the `task` variable in `main.py` to either "A" for PneumoniaMNIST or "B" for PathMNIST.
2. Set the `model` variable to "SVM", "CNN", or "Random Forest" depending on the model you wish to use.
3. Optionally, set the `retrain` flag to `True` if you want to retrain the model, or `False` to use a pre-trained model.

## Requirements

To run this project, your environment need to include Python 3.8 and several packages. The `environment.yml` file in this repository lists all the necessary dependencies.
The project requires the following packages:
- numpy
- torch
- joblib
- medmnist
- sklearn
- matplotlib

## Notes
- Ensure all the paths and file names are correct and match your project directory structure.
- Adjust the README as needed to reflect any changes or additions to your project.
