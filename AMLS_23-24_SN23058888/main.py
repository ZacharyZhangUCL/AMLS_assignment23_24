import os
from medmnist import PneumoniaMNIST  # A
from medmnist import PathMNIST  # B
import numpy as np
import torch
import joblib
from A.A_SVM import A_SVM_training, A_SVM_testing
from A.A_CNN import A_CNN_training, A_CNN_testing
from B.B_RandomForest import B_RandomForest_training, B_RandomForest_testing
from B.B_CNN import B_CNN_training, B_CNN_testing


def get_data(task:str):

    if task == "A":
        dataset = "PneumoniaMNIST"
    elif task == "B":
        dataset = "PathMNIST"
    else:
        return

    # Get the full path to the current file
    file_path = os.path.abspath(__file__)
    root_path = os.path.dirname(file_path)

    datasets_path = os.path.join(root_path, "Datasets")
    dataset_task_path = os.path.join(datasets_path, dataset)

    # If it doesn't exist, create a directory
    if not os.path.exists(dataset_task_path):
        os.makedirs(dataset_task_path)
    
    if task == "A":
        dataset_dataset_task_npz = os.path.join(dataset_task_path, 'pneumoniamnist.npz')
        if not os.path.exists(dataset_dataset_task_npz):
            dataset = PneumoniaMNIST(split="train", download=True, root=dataset_task_path)
        data = np.load(dataset_dataset_task_npz)
    else:
        dataset_dataset_task_npz = os.path.join(dataset_task_path, 'pathmnist.npz')
        if not os.path.exists(dataset_dataset_task_npz):
            dataset = PathMNIST(split="train", download=True, root=dataset_task_path)
        data = np.load(dataset_dataset_task_npz)
    return data

if __name__ == "__main__":
    task = "B"  # "A" or "B"
    model = "Random Forest"  # "SVM", "CNN" or "Random Forest"
    retrain = False  # True (Retrain the model) or False (Load the existing model)

    data = get_data(task=task)
    train_images = data['train_images']
    train_labels = data['train_labels']
    val_images = data['val_images']
    val_labels = data['val_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    if task == "A" and model == "SVM":
        # Train and Valid
        if retrain == True:
            model_A_SVM = A_SVM_training(train_images, train_labels, val_images, val_labels)
        else:
            model_A_SVM = joblib.load('A/A_SVM_model.joblib')
        # Test
        A_SVM_testing(model_A_SVM, test_images, test_labels)

    elif task == "A" and model == "CNN":
        # Train and Valid
        if retrain == True:
            model_A_CNN = A_CNN_training(train_images, train_labels, val_images, val_labels)
        else:
            model_A_CNN = torch.load('A/A_CNN_model.pth')
        # Test
        A_CNN_testing(model_A_CNN, test_images, test_labels)
    
    elif task == "B" and model == "Random Forest":
        # Train and Valid
        if retrain == True:
            model_B_RandomForest = B_RandomForest_training(train_images, train_labels, val_images, val_labels)
        else:
            model_B_RandomForest = joblib.load('B/B_RandomForest_model.joblib')
        # Test
        B_RandomForest_testing(model_B_RandomForest, test_images, test_labels)

    elif task == "B" and model == "CNN":
        # Train and Valid
        if retrain == True:
            model_B_CNN = B_CNN_training(train_images, train_labels, val_images, val_labels)
        else:
            model_B_CNN = torch.load('B/B_CNN_model.pth')
        # Test
        B_CNN_testing(model_B_CNN, test_images, test_labels)

    else:
        print("Errors, Tasks and Methods do not Meet Requirements!!!")