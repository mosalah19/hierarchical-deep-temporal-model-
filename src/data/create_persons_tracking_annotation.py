import pickle
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# in volleyball_tracking_annotation contain all videos that contail all mainframe that contain annotation for each payer
Video_folder_path = r"D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\volleyball_tracking_annotation"
list_videos = [f'{i}' for i in range(0, 55)]
annotation = {'video': [], 'MainFrame': [], 'position': [], 'label': []}
for video in list_videos:
    videos_path = rf'{Video_folder_path}\{video}'
    for clip in os.listdir(videos_path):
        with open(rf'{videos_path}\{clip}\{clip}.txt', mode='r') as file_object:
            lines = file_object.readlines()
            for line in range(10, len(lines), 20):
                list_data = lines[line].split(" ")
                MainFrame = list_data[5]
                label = list_data[9]
                position = list_data[1:5]
                annotation['position'].append(position)
                annotation['video'].append(video)
                annotation['MainFrame'].append(MainFrame)
                annotation['label'].append(label)

annotation_df = pd.DataFrame(annotation)
label_encoder = LabelEncoder()

# Perform label encoding
annotation_df['label'] = label_encoder.fit_transform(annotation_df['label'])

# save in pkl file
with open('persons_tracking_annotation.pkl', 'wb') as f:
    pickle.dump(annotation_df, f)
