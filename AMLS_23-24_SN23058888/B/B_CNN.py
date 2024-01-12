import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# Preprocessing for CNN
class preprocessing_CNN(Dataset):
    def __init__(self, images, labels, train=True):
        self.num_augment = 3
        self.images = images
        self.labels = labels.ravel()
        self.train = train

    def __len__(self):
        # Returns the size of the dataset, with the training set containing data augmentation and the test set not containing data augmentation
        return len(self.images) * self.num_augment if self.train == True else len(self.images)

    def __getitem__(self, index):
        # Determine the original image index
        original_index = index // self.num_augment if self.train == True else index

        image = self.images[original_index]
        label = self.labels[original_index]
        image = image.transpose(2, 0, 1)
        
        if self.train == True:
            # Determine the data augmentation image index
            image_variation = index % self.num_augment
            # Noise Adding
            image = self.add_noise(image)
            # Data Augmentation
            image = self.apply_augmentation(image, image_variation)

        # Normalize the images to be in the range [0, 1]
        image = image.astype('float32') / 255.0

        # Convert Data to Tensors
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return image, label
    
    def apply_augmentation(self, image, image_variation):
        # Apply different enhancements depending on the variation index
        if image_variation == 0:
            # Original Image
            pass
        elif image_variation == 1:
            # First Augmentation -> Image Flipping
            image = self.flip(image)
        elif image_variation == 2:
            # Second Augmentation -> Image Rotating
            image = self.rotate(image)
        return image
    
    # Noise Adding
    def add_noise(self, image, mean=0, std=5):
        height, width = image.shape[-2:]
        gauss = np.random.normal(mean, std, (width, height))
        image = np.clip(image + gauss, 0, 255).astype(np.uint8)
        return image

    # Image Flipping
    def flip(self, image):
        flip_type = random.choice(['horizontal', 'vertical'])
        if flip_type == 'horizontal':
            # Flip the image horizontally
            return image[:, ::-1]
        else:
            # Flip the image vertically
            return image[::-1, :]
    
    # Image Rotating
    def rotate(self, image):
        # Corresponding to 90, 180, 270 degrees
        angle = random.choice([1, 2, 3])  # Corresponding to 90, 180, 270 degrees

        # Rotate the image by the selected angle
        image = np.rot90(image, angle, axes=(1, 2))

        return image


# CNN Model Definition
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layer
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=num_classes)

    def forward(self, x):
        # Pass through Layer 1
        x = self.maxpool1(self.relu1(self.conv1(x)))
        
        # Pass through Layer 2
        x = self.maxpool2(self.relu2(self.conv2(x)))
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc1(x)
        
        return x


# Early Stopping
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience  # The number of waiting periods
        self.min_delta = min_delta  # Minimal improvement
        self.counter = 0  # A counter that tracks the number of cycles since the last improvement
        self.best_loss = None  # Record the optimal loss value
        self.early_stop = False  # Whether or not to trigger an early stop

    def __call__(self, val_loss):
        if self.best_loss is None:
            # If it is the first period, the optimal loss is directly set to the current loss
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            # If the current loss is not lower than the optimal loss (taking into account min_delta), increase the counter
            self.counter += 1
            if self.counter >= self.patience:
                # If the counter reaches the waiting period, an early stop is triggered
                self.early_stop = True
        else:
            # If the current loss improves, update the best loss and reset the counter
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


# Train the CNN classifier and Evaluate the model
def B_CNN_training(train_images, train_labels, val_images, val_labels,
          num_epochs = 100, # Number of training epochs
          learning_rate = 1e-3, # Learning rate
          batch_size = 64, # Batch size
          num_classes = 9, # Number of output classes
          ):

    # Hyperparameters
    num_epochs = num_epochs # Number of training epochs
    learning_rate = learning_rate # Learning rate
    batch_size = batch_size # Batch size
    num_classes = num_classes # Number of output classes (normal or pneumonia)
    
    # Inputs Preprocessing
    train_dataset = preprocessing_CNN(train_images, train_labels, train=True)
    val_dataset = preprocessing_CNN(val_images, val_labels, train=False)

    # Data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, drop_last=False)

    # Model, Loss, Optimizer, Scheduler and Early_stopper
    model = CNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-7)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, min_lr=learning_rate*1e-3)
    early_stopper = EarlyStopping(patience=30)

    # Initialize lists to store Loss and Accuracy Score
    train_loss = []
    train_accuracy_score = []
    val_loss = []
    val_accuracy_score = []

    # Training and Validation Loop
    for epoch in range(num_epochs):
        # Initialize lists to store true labels and predictions
        train_true_labels = []
        train_predictions = []
        total_train_loss = 0

        # Training Loop
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Get the predicted class with the highest score
            _, predicted = torch.max(outputs.data, 1)
            
            # Append actual and predicted labels to lists
            train_true_labels += labels.tolist()
            train_predictions += predicted.tolist()

        # Append train_loss and accuracy_score to lists
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_accuracy_score = accuracy_score(train_true_labels, train_predictions)
        train_loss.append(avg_train_loss)
        train_accuracy_score.append(avg_train_accuracy_score)
        
        # Set the model to evaluation mode
        model.eval()

        # Initialize lists to store true labels and predictions
        val_true_labels = []
        val_predictions = []
        total_val_loss = 0

        # No gradient is needed for evaluation
        with torch.no_grad():
            # Validation Loop
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                # Get the predicted class with the highest score
                _, predicted = torch.max(outputs.data, 1)
                
                # Append actual and predicted labels to lists
                val_true_labels += labels.tolist()
                val_predictions += predicted.tolist()
        
        # Append Loss and Accuracy Score to lists
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_accuracy_score = accuracy_score(val_true_labels, val_predictions)
        val_loss.append(avg_val_loss)
        val_accuracy_score.append(avg_val_accuracy_score)

        # Use scheduler.step()
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1:{len(str(num_epochs))}d}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy Score: {avg_train_accuracy_score:.4f}, Valid Loss: {avg_val_loss:.4f}, Valid Accuracy Score: {avg_val_accuracy_score:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.1E}")


        if early_stopper(avg_val_loss):
            print("Early Stopping!")
            break
    
    # Gets the root_path where the file is located
    file_path = os.path.abspath(__file__)
    root_path = os.path.dirname(file_path)

    # Saving the entire model  
    torch.save(model, os.path.join(root_path, "B_CNN_model.pth"))

    # Plot the training and testing loss and accuracy on the same graph using dual axes
    plt.figure(figsize=(10, 5))

    # Create the first axis for the loss
    ax1 = plt.gca() # get current axis
    loss_line1 = ax1.plot(train_loss, label='Train Loss', color='red', linestyle='dashed')[0]
    loss_line2 = ax1.plot(val_loss, label='Valid Loss', color='red', linestyle='solid')[0]
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_ylim(0, 0.5)  # Setting the y-limit for ax1

    # Create a second axis for the accuracy
    ax2 = ax1.twinx() # create a twin of the first axis
    accuracy_line1 = ax2.plot(train_accuracy_score, label='Train Accuracy Score', color='blue', linestyle='dashed')[0]
    accuracy_line2 = ax2.plot(val_accuracy_score, label='Valid Accuracy Score', color='blue', linestyle='solid')[0]
    ax2.set_ylabel('Accuracy Score', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0.5, 1)  # Setting the y-limit for ax2
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))  # Format the second axis labels as percentages

    plt.title('Training and Validation Loss and Accuracy Score')

    # Creating a combined legend
    lines = [loss_line1, loss_line2, accuracy_line1, accuracy_line2]
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc='center right') 

    # Saves the plot as a PNG file
    plt.savefig(os.path.join(root_path, "B_CNN_trainging_plot.png"))

    return model

# Test the CNN classifier
def B_CNN_testing(model, test_images, test_labels):
    # Inputs Preprocessing
    test_dataset = preprocessing_CNN(test_images, test_labels, train=False)

    # Data loaders
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False)

    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to store true labels and predictions
    test_true_labels = []
    test_predictions = []

    # No gradient is needed for evaluation
    with torch.no_grad():
        # Testing Loop
        for images, labels in test_loader:
            outputs = model(images)
            
            # Get the predicted class with the highest score
            _, predicted = torch.max(outputs.data, 1)
            
            # Append actual and predicted labels to lists
            test_true_labels += labels.tolist()
            test_predictions += predicted.tolist()
    
    # Append train_loss and accuracy_score to lists
    print(f"Accuracy Score: {accuracy_score(test_true_labels, test_predictions)}")
    print(f"Classification Report:\n{classification_report(test_true_labels, test_predictions)}")

    return
