"""

 ●For each clip, use the middle image only
   ○ Fee free to use 5 before and 4 after also
 ● Fine-tune an image classifier over 8 classes
 ● Compute the results. This is your first mode

"""
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from train_model import *
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import importlib.util
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
    N_CLASS = 8
    LR = 0.01
    EPOCS = 10
    PRINT_FREQUANCE = 100

    # import dataset and split it to train and  test
    '''
        - The dataset is 55 videos. Each video has a directory for it with sequntial IDs (0, 1...54)
        - Train Videos: 1 3 6 7 10 13 15 16 18 22 23 31 32 36 38 39 40 41 42 48 50 52 53 54 4 5 9 11 14 20 21 25 29 34 35 37 43 44 45 47
        - Validation Videos: 0 2 8 12 17 19 24 26 27 28 30 33 46 49 51
    '''

    # read csv file contain three column video Number and Mainframe and class
    csv_path = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\annotations\video_annotation.csv'
    df_dataset = pd.read_csv(csv_path)

    # split it to train and  test
    training_set = df_dataset[df_dataset['video'].isin([
        1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54, 4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47])]
    Validation_set = df_dataset[df_dataset['video'].isin([
        0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51])]

    # configration data_loader
    # source path (path of videos folder that contain all videos)
    Video_folder_path = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\videos'

    # transormation for train
    transformation = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop((224, 224)),
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
    trainig = annotation_Loader.Video_annotation_loader(
        Video_folder_path, training_set, transformation)
    val = annotation_Loader.Video_annotation_loader(
        Video_folder_path, Validation_set, vtransformation)

    # create train and val loder
    training_loader = DataLoader(dataset=trainig, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val, batch_size=32, shuffle=False)

    # configuration model

    # model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # tuning model
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features=2048,
                  out_features=N_CLASS, bias=True),

    )
    # frozen backpone
    for child in list(model.children())[:-3]:
        for param in child.parameters():
            param.requires_grad = False

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001)

    # Train the model
    train(model=model, train_dataloader=training_loader, test_dataloader=val_loader,
          loss_fn=criterion, optimizer=optimizer, epochs=EPOCS, print_freq=PRINT_FREQUANCE)
