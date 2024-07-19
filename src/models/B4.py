# Import necessary libraries
import torch
import importlib
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import pandas as pd
from train_model import *

# work with GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Specify the path to the Python file
file_path = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\src\data\volleyball_annotation_Loader.py'

# Create a spec object using the file path
spec = importlib.util.spec_from_file_location("module_name", file_path)

# Load the module using the spec
annotation_Loader = importlib.util.module_from_spec(spec)

# Execute the module
spec.loader.exec_module(annotation_Loader)


# Encoder_nn class to take images and represent each image as a feature vector using a pre-trained ResNet50 model
class Encoder_nn(nn.Module):
    def __init__(self):
        super().__init__()

        # Load the pre-trained ResNet50 model with default weights
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Define a ReLU activation function
        self.relu = nn.ReLU()

        # Define a dropout layer with a dropout probability of 0.5
        self.dropout = nn.Dropout(0.5)

        # Freeze the parameters of the pre-trained ResNet50 model to prevent training
        for p in self.model.parameters():
            p.requires_grad = False

        # Replace the fully connected layer with an identity function to output the feature vector directly
        self.model.fc = nn.Identity()

    def forward(self, x):
        # Pass the input through the ResNet50 model to get the feature vector
        x = self.model(x)

        # Apply ReLU activation and dropout to the feature vector
        y = self.dropout(self.relu(x))

        # Return the processed feature vector
        return y


# Decoder_lstm class to decode feature vectors into class predictions using an LSTM and a fully connected layer
class Decoder_lstm(nn.Module):
    def __init__(self):
        super().__init__()

        # Define an LSTM layer with input size 2048 and hidden size 1024, 1 layer, with batch_first=True
        self.lstm = nn.LSTM(2048, 1024, num_layers=1, batch_first=True)

        # Define a fully connected layer to map LSTM output to 8 classes
        self.fc = nn.Linear(1024, 8)

        # Define a ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass the input through the LSTM layer
        out, _ = self.lstm(x)

        # Select the last time step's output
        out = out[:, -1, :]

        # Apply the fully connected layer and ReLU activation
        y = self.relu(self.fc(out))

        # Return the final output
        return y


# Define the combined model class
class EncoderToDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize the encoder and decoder components
        self.encoder = Encoder_nn()
        self.decoder = Decoder_lstm()

    def forward(self, x):
        # Reshape input to the expected shape for the encoder (batch_size * sequence_length, channels, height, width)
        x = x.view(-1, 3, 224, 224)

        # Pass the reshaped input through the encoder to get feature vectors
        features = self.encoder(x)

        # Reshape feature vectors to the expected shape for the decoder (batch_size, sequence_length, feature_size)
        # using sequence_length = 9 because I used Use 9 frames per image
        features = features.view(-1, 9, 2048)

        # Pass the feature vectors through the decoder to get the final output
        y = self.decoder(features)

        # Return the final output
        return y


if __name__ == "__main__":
   # Initialize the model
    model = EncoderToDecoder()

    # Define the path to the CSV file containing video annotations
    csv_path = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\annotations\video_annotation.csv'

    # Load the dataset from the CSV file
    df_dataset = pd.read_csv(csv_path)

    # Split the dataset into training, validation, and test sets based on video IDs
    training_set = df_dataset[df_dataset['video'].isin([
        1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54, 4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47])]
    Validation_set = df_dataset[df_dataset['video'].isin([
        0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51])]
    test_set = df_dataset[df_dataset['video'].isin([
        4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47])]

    # Define the path to the video folder
    Video_folder_path = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\videos'

    # Define image transformations
    transformation = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    # Load the training and test datasets using custom data loaders
    train_dataset = annotation_Loader.Video_Sequance_annotation_loader(
        Video_folder_path, training_set, transformation)
    test_dataset = annotation_Loader.Video_Sequance_annotation_loader(
        Video_folder_path, Validation_set, transformation)

    # Configure the data loaders with batch size and shuffling
    training_loader = DataLoader(
        dataset=train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=32, shuffle=False)

    # Define the loss function
    loss_fun = nn.CrossEntropyLoss()

    # Define the optimizer (Adam in this case)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model with the specified parameters
    train(model=model,
          train_dataloader=training_loader,
          test_dataloader=test_loader,
          loss_fn=loss_fun,
          save_name="B4",
          optimizer=optimizer,
          epochs=10,
          print_freq=50)
