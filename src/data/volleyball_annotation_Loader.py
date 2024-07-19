from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import environ
import ast
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2


def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"


class Video_annotation_loader(Dataset):
    def __init__(self, src_folder, annotation_file, transformation=None):
        super().__init__()
        suppress_qt_warnings()
        self.src_folder = src_folder
        self.annotation_file = annotation_file
        self.transformation = transformation

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, index):
        video, MainFrame = self.annotation_file.iloc[index][[
            'video', 'MainFrame']]
        image_path = f"{self.src_folder}\{video}\{MainFrame[:-4]}\{MainFrame}"
        image = plt.imread(image_path)
        y_label = self.annotation_file.iloc[index]['label']
        if self.transformation:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8))
            image = self.transformation(image)
        return [image, y_label]


class person_tracking_annotation_loader(Dataset):
    def __init__(self, src_folder, annotation_file, transformation=None):
        super().__init__()
        suppress_qt_warnings()
        self.src_folder = src_folder
        self.annotation_file = annotation_file
        self.transformation = transformation

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, index):
        video, MainFrame = self.annotation_file.iloc[index][[
            'video', 'MainFrame']]
        image_path = f"{self.src_folder}\{video}\{MainFrame}\{MainFrame}.jpg"
        # image = plt.imread(image_path)
        image = Image.open(image_path).convert('RGB')
        # Convert the string to a list
        # list_of_strings = ast.literal_eval(
        #     self.annotation_file.iloc[index]['position'])
        # list_of_integars = [int(x) for x in list_of_strings]
        xmin, ymin, xmax, ymax = self.annotation_file.iloc[index]['position']
        cropped_image = image.crop(
            (int(xmin), int(ymin), int(xmax), int(ymax)))
        # cropped_image = image[ymin:ymax, xmin:xmax]
        y_label = self.annotation_file.iloc[index]['label']
        if self.transformation:
            if isinstance(cropped_image, np.ndarray):
                cropped_image = Image.fromarray(cropped_image.astype(np.uint8))
            cropped_image = self.transformation(cropped_image)
        return [cropped_image, y_label]


class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = csv_file
        # Convert 'feature_vector' column from strings to numpy arrays

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get feature vector and label for the given index
        feature_vector = self.data.iloc[idx]['feature_vector']
        label = self.data.iloc[idx]['label']
        # Convert feature vector and label to PyTorch tensors
        feature_vector = torch.tensor(feature_vector, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return feature_vector, label


class Video_Sequance_annotation_loader(Dataset):
    def __init__(self, src_folder, annotation_file, transformation=None):
        super().__init__()
        self.src_folder = src_folder
        self.annotation_file = annotation_file
        self.transformation = transformation

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, index):
        video, MainFrame = self.annotation_file.iloc[index][[
            'video', 'MainFrame']]
        images = []
        for i in range(int(MainFrame[:-4])-4, int(MainFrame[:-4])+5):
            image_path = f"{self.src_folder}\{video}\{MainFrame[:-4]}\{str(i)}.jpg"
            image = Image.open(image_path)
            if self.transformation:
                image = self.transformation(image)
            images.append(image)
        images = torch.stack(images)
        y_label = self.annotation_file.iloc[index]['label']

        return images, y_label


class person_Sequance_annotation_loader(Dataset):
    def __init__(self, src_folder, annotation_file, transformation=None):
        super().__init__()
        self.src_folder = src_folder
        self.annotation_file = annotation_file
        self.transformation = transformation

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, index):
        video, MainFrame = self.annotation_file.iloc[index][[
            'video', 'MainFrame']]
        images = []
        for i in range(MainFrame-4, MainFrame+5):
            image_path = f"{self.src_folder}\{video}\{MainFrame}\{str(i)}.jpg"
            image = Image.open(image_path)
            xmin, ymin, xmax, ymax = self.annotation_file.iloc[index]['position']
            cropped_image = image.crop(
                (int(xmin), int(ymin), int(xmax), int(ymax)))
            if self.transformation:
                cropped_image = self.transformation(cropped_image)
            images.append(cropped_image)
        images = torch.stack(images)
        y_label = self.annotation_file.iloc[index]['label']

        return images, y_label

# if __name__ == "__main__":
#     train_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
#         #     0.229, 0.224, 0.225]),
#     ])
#     Video_folder_path = r"D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\videos"
#     annotation_df = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\annotations\persons_tracking_annotation.csv'
#     annotation_df = pd.read_csv(annotation_df)
#     x = person_tracking_annotation_loader(
#         Video_folder_path, annotation_df, train_transform)
#     training_data = DataLoader(dataset=x, batch_size=11, shuffle=False)
#     out = next(iter(training_data))
#     title = out[1][0]
#     image_tensor = out[0][1]
#     image_np = image_tensor.permute(1, 2, 0).numpy()
#     cv2.imshow('Cropped Image', np.array(image_np))
#     cv2.waitKey(0)
#     # clip_dir_path = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data/videos\0\13286'
#     # img_path = os.path.join(clip_dir_path, f'13286.jpg')
#     # image = Image.open(img_path).convert('RGB')
#     # x1, y1, x2, y2 = [90, 452, 157, 586]
#     # cropped_image = image.crop((x1, y1, x2, y2))
#     # cropped_image = train_transform(cropped_image)
#     # cv2.imshow('Cropped Image',
#     #            np.array(cropped_image))
#     # cv2.waitKey(0)
