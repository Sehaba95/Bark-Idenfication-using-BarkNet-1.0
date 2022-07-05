from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import torch

class TestDataset(Dataset):
    def __init__(self, csv_path, dataset_path, input_size):
        self.data_info = pd.read_csv(csv_path, index_col=False, usecols=['Image path', 'Directory', 'Class ID', 'Class name']) # Read the csv file
        self.image_arr = np.asarray(self.data_info.iloc[:, 0]) # First column contains the image paths
        self.label_arr = np.asarray(self.data_info.iloc[:, 2]) # Third column is the labels
        self.data_len = len(self.data_info.index)
        self.dataset_path = dataset_path
        self.transform = transforms.Compose([  
                                                transforms.RandomCrop(input_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        image_name = self.image_arr[idx] # Get image name from the pandas df
        image = Image.open("".join([self.dataset_path, image_name])).convert('RGB') # Open image
        image = self.transform(image)
        label = torch.tensor(self.label_arr[idx]) # Get label(class) of the image based on the cropped pandas column

        return image, label

    def __len__(self):
        return self.data_len


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def get_test_dataloader(dataset_path, batch_size, input_size):
    test_set_path = "BarkNet_1_0/barknet_test.csv"
    test_data = TestDataset(test_set_path, dataset_path, input_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True,
                                               num_workers=6, drop_last=True, pin_memory=True)
    return test_loader