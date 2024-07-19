from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

Video_folder_path = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\videos'
list_videos = [f'{i}' for i in range(0, 55)]

annotation = {'video': [], 'MainFrame': [], 'label': []}
for video in list_videos:
    with open(rf'{Video_folder_path}\{video}\annotations.txt', mode='r') as file_object:
        data = file_object.readlines()
        for frame in data:
            MainFrame, label = frame.split(" ")[:2]
            annotation['video'].append(video)
            annotation['MainFrame'].append(MainFrame)
            annotation['label'].append(label)

annotation_df = pd.DataFrame(annotation)


# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Perform label encoding
annotation_df['label'] = label_encoder.fit_transform(annotation_df['label'])
# save annotation  in pkl file
with open('video_annotation.pkl', 'wb') as f:
    pickle.dump(annotation_df, f)

# annotation_df.to_csv(
#     r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\annotations\video_annotation.csv', index=False)
