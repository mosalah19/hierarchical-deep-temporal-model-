
import importlib
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import pickle
# Specify the path to the Python file
file_path = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\src\data\volleyball_annotation_Loader.py'

# Create a spec object using the file path
spec = importlib.util.spec_from_file_location("module_name", file_path)

# Load the module using the spec
annotation_Loader = importlib.util.module_from_spec(spec)

# Execute the module
spec.loader.exec_module(annotation_Loader)

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    # make dictionary  to save video and main frame and max pool of all player in main frame
    result = {"video": [], "MainFrame": [], "feature_vector": []}

    # load person annotation that contain (video , main frame , personal position , label)
    with open("persons_tracking_annotation.pkl", 'rb')as f:
        df_person_tracking = pd.DataFrame(pickle.load(f))

    # load video annotation that contain (video , main frame  , label)
    Video_folder_path = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\videos'

    # transformation
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    # configuration model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Flatten()
    )
    model.to(device)

    '''
   Explanation
    We need to process each main frame of a video, sort the detected persons in each frame based on their positioning information,
    and then create batches of size 6 without shuffling. This batch size is chosen because each main frame contains
    12 players: 6 in the left team and 6 in the right team. 
    We want to perform max pooling for each team individually and then combine the results.
    
    '''

    # List of videos to process
    list_videos = [i for i in range(0, 55)]

    for l in list_videos:
        # Filter dataframe for the current video
        x = df_person_tracking[df_person_tracking['video'] == str(l)]

        for j in list(x['MainFrame'].unique()):
            # Create a copy of the dataframe filtered by the current MainFrame
            df_copy = x.copy()
            df_copy = df_copy[df_copy["MainFrame"] == j]

            # Extract the x-coordinate from the position and add it as a column
            df_copy['x'] = df_copy['position'].apply(lambda l: int(l[0]))

            # Sort the dataframe by x-coordinate in ascending order
            df_copy.sort_values(by="x", ascending=True, inplace=True)

            # Load the dataset for the current MainFrame
            dataset = annotation_Loader.person_tracking_annotation_loader(
                Video_folder_path, df_copy, transformation)

            # Create a DataLoader with batch size 6 and no shuffling
            loader = DataLoader(dataset=dataset, batch_size=6, shuffle=False)

            model.eval()
            # Initialize lists to store results
            dic = []
            numpy_array = []

            # Disable gradient calculation for inference
            with torch.inference_mode():
                # Iterate over batches in the DataLoader
                for i, (image, label) in enumerate(loader, 1):
                    # Move image to the appropriate device
                    image = image.to(device)

                    # Get feature vector from the model
                    feature_vector = model(image)

                    # Append the feature vector to the list
                    numpy_array.append(feature_vector.to('cpu').numpy())

            # Perform max pooling on the feature vectors for each group
            max_pooled_vector_G0 = np.max(numpy_array[0], axis=0)
            max_pooled_vector_G1 = np.max(numpy_array[1], axis=0)

            # Concatenate the max pooled vectors from both groups
            max_pooled_vector = np.concatenate(
                [max_pooled_vector_G0, max_pooled_vector_G1], axis=0)

            # Append the results to the result dictionary
            result['video'].append(l)
            result["MainFrame"].append(j)
            result["feature_vector"].append(max_pooled_vector)

    # save result in pickle file
    with open('feature_vectors.pkl', 'wb') as f:
        pickle.dump(result, f)*()
