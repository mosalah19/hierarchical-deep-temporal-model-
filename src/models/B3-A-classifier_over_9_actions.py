import importlib
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from train_model import *
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import torch


# Specify the path to the Python file
file_path = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\src\data\volleyball_annotation_Loader.py'

# Create a spec object using the file path
spec = importlib.util.spec_from_file_location("module_name", file_path)

# Load the module using the spec
annotation_Loader = importlib.util.module_from_spec(spec)

# Execute the module
spec.loader.exec_module(annotation_Loader)


if __name__ == '__main__':
    N_CLASS = 9
    LR = 0.01
    EPOCS = 10
    PRINT_FREQUANCE = 100
    # import dataset and split it to train and validation and test
    '''
        - The dataset is 55 videos. Each video has a directory for it with sequntial IDs (0, 1...54)
        - Train Videos: 1 3 6 7 10 13 15 16 18 22 23 31 32 36 38 39 40 41 42 48 50 52 53 54 4 5 9 11 14 20 21 25 29 34 35 37 43 44 45 47
        - Validation Videos: 0 2 8 12 17 19 24 26 27 28 30 33 46 49 51
    '''
    # read csv file contain video_number , image of each player of each mainframe , positionof player in mainframe , label

    csv_path = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\annotations\persons_tracking_annotation.csv'
    df_dataset = pd.read_csv(csv_path)
    # split it to train and  test
    training_set = df_dataset[df_dataset['video'].isin([
        1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39,
        40, 41, 42, 48, 50, 52, 53, 54, 4, 5, 9, 11, 14, 20, 21, 25,
        29, 34, 35, 37, 43, 44, 45, 47])]
    Validation_set = df_dataset[df_dataset['video'].isin([
        0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51])]

    # configration data_loader
    # source path (path of videos folder that contain all videos)

    # transormation for train
    Video_folder_path = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\videos'
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    # transormation for val
    vtransformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    # annotation_Loader
    trainig = annotation_Loader.person_tracking_annotation_loader(
        Video_folder_path, training_set, transformation)
    val = annotation_Loader.person_tracking_annotation_loader(
        Video_folder_path, Validation_set, vtransformation)

    # create train and val loder
    training_loader = DataLoader(dataset=trainig, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val, batch_size=32, shuffle=False)

    # configuration model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features=2048,
                  out_features=N_CLASS, bias=True),

    )
    for child in list(model.children())[:-3]:
        for param in child.parameters():
            param.requires_grad = False

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR)

    # Train the model
    train(model=model, train_dataloader=training_loader, test_dataloader=val_loader,
          loss_fn=criterion, optimizer=optimizer, epochs=EPOCS, print_freq=PRINT_FREQUANCE)
