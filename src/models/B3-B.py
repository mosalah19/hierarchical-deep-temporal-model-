import importlib
import torch.nn as nn
from torch import nn
import pandas as pd
import torch
from train_model import *
from torch.utils.data import DataLoader
import pickle

# Specify the path to the Python file
file_path = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\src\data\volleyball_annotation_Loader.py'

# Create a spec object using the file path
spec = importlib.util.spec_from_file_location("module_name", file_path)

# Load the module using the spec
annotation_Loader = importlib.util.module_from_spec(spec)

# Execute the module
spec.loader.exec_module(annotation_Loader)


class Classifier(nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Linear(n_in_feature, n_out_feature),
        )

    def forward(self, x):
        y = self.layer_1(x)
        return y


if __name__ == '__main__':
    N_CLASS = 8

    # read videos annotation that contain (video,MainFrame,label)
    df_video_annotation = pd.read_csv(
        r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\annotations\video_annotation.csv')

    # Load the dictionary that contain (video, MainFrame , feature_vector) from the pickle file
    with open('feature_vectors.pkl', 'rb') as f:
        df_feature_vector = pd.DataFrame(pickle.load(f))

    # add '.jpg' to main frame
    df_feature_vector['MainFrame'] = df_feature_vector['MainFrame'].astype(
        str) + '.jpg'

    # merge video annotation and feature vactor to get label for each feature vector
    df_feature_vector_with_label = df_video_annotation.merge(
        df_feature_vector, on=["MainFrame", "video"], how='inner')

    # split it to train and  test
    training_set = df_feature_vector_with_label[df_feature_vector_with_label['video'].isin([
        1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54,
        4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47])]
    Validation_set = df_feature_vector_with_label[df_feature_vector_with_label['video'].isin([
        0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51])]

    #  simple ANN
    nn_model = Classifier(4096, N_CLASS)

    training = annotation_Loader.CustomDataset(training_set)
    Validation = annotation_Loader.CustomDataset(Validation_set)

    # create train and val loader
    training_loader = DataLoader(
        dataset=training, batch_size=32, shuffle=False)
    val_loader = DataLoader(dataset=Validation, batch_size=64, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        nn_model.parameters(), lr=0.0001, weight_decay=1e-4)

    # Train the model
    train(model=nn_model, train_dataloader=training_loader, test_dataloader=val_loader,
          loss_fn=criterion, optimizer=optimizer, epochs=25, print_freq=1000)
# Define loss function and optimizer
