import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms
import pandas as pd
from train_model import *
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"


class person_annotation_loader(Dataset):
    def __init__(self, src_folder, person_annotation_file, video_annotation_file, transformation=None):
        super().__init__()
        self.src_folder = src_folder
        self.person_annotation_file = person_annotation_file
        self.video_annotation_file = video_annotation_file
        self.transformation = transformation

    def __len__(self):
        return len(self.person_annotation_file)

    def __getitem__(self, index):
        video, MainFrame = self.person_annotation_file.iloc[index][[
            'video', 'MainFrame']]
        images = []
        for i in range(MainFrame-4, MainFrame+5):
            image_path = f"{self.src_folder}\{video}\{MainFrame}\{str(i)}.jpg"
            image = Image.open(image_path)
            xmin, ymin, xmax, ymax = self.person_annotation_file.iloc[index]['position']
            cropped_image = image.crop(
                (int(xmin), int(ymin), int(xmax), int(ymax)))
            if self.transformation:
                cropped_image = self.transformation(cropped_image)
            images.append(cropped_image)
        images = torch.stack(images)
        y_label = self.video_annotation_file[
            (self.video_annotation_file["video"] == video) &
            (self.video_annotation_file["MainFrame"] == f"{MainFrame}.jpg")]['label'].item()

        return images, y_label


class Encoder_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.fc = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        y = self.dropout(self.relu(x))
        return y


class Decoder_lstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2048, 1024, num_layers=1, batch_first=True)
        self.fc = nn.Linear(2048, 8)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        tensor_list = []
        for i in range(1, int((out.shape[0]/12))+1):
            max_pooled_vector_G0 = torch.max(out[:(i*6)], dim=0).values
            max_pooled_vector_G1 = torch.max(
                out[(i*6):((i+1)*6)], dim=0).values
            max_pooled_vector = torch.concatenate(
                [max_pooled_vector_G0, max_pooled_vector_G1], dim=0).to(device)
            tensor_list.append(max_pooled_vector)
        out = torch.stack(tensor_list)
        y = self.relu(self.fc(out))
        return y


class EncoderToDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder_nn()

        self.decoder = Decoder_lstm()

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        features = self.encoder(x)
        features = features.view(-1, 9, 2048)
        y = self.decoder(features)
        return y


if __name__ == "__main__":
    with open(r"D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\annotations\persons_tracking_annotation.pkl", 'rb') as f:
        df_person_tracking = pd.DataFrame(pickle.load(f))

    with open(r"D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\annotations\video_annotation.pkl", 'rb') as f:
        df_video_annotation = pd.DataFrame(pickle.load(f))

    df_person_tracking['x'] = df_person_tracking['position'].apply(
        lambda l: int(l[0]))
    df_person_tracking["MainFrame"] = df_person_tracking["MainFrame"].astype(
        'int64')
    df_person_tracking.sort_values(
        by=["video", "MainFrame", "x"], ascending=True, inplace=True)
    # print(df_person_tracking)
    Video_folder_path = r'D:\Data Science\Final_Project\hierarchical_deep_temporal_models_for_group_activity_recognition\data\videos'
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    x = person_annotation_loader(
        Video_folder_path, df_person_tracking, df_video_annotation, transformation)
    loder = DataLoader(x, batch_size=36, shuffle=False)
    # Create an iterator from the DataLoader
    data_iter = iter(loder)

    # Get the next batch of data
    images, labels = next(data_iter)

    # Print the batch of images and labels
    print(images.shape)
    print(labels.shape)

    # model = EncoderToDecoder()
    # model.to(device)
    # model.eval()
    # with torch.inference_mode():
    #     for inputs, targets in loder:
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = model(inputs)
    #         print(outputs.shape)
