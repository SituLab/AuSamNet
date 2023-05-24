import os
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from utils import ImageTransforms,RGB2Bayer

class Dataset_Bayer(Dataset):
    def __init__(self, data_folder, split, hr_img_type, train_data_name=None ,test_data_name=None, verify_data_name=None):
        self.data_folder = data_folder
        self.split = split.lower()
        self.hr_img_type = hr_img_type
        self.train_data_name = train_data_name
        self.test_data_name = test_data_name
        self.verify_data_name = verify_data_name

        assert hr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}

        assert self.split in {'train', 'test', 'verify'}
        if self.split == 'test' and self.test_data_name is None:
            raise ValueError("Please provide the name of the test dataset!")
        if self.split == 'train' and self.train_data_name is None:
            raise ValueError("Please provide the name of the train dataset!")
        if self.split == 'verify' and self.verify_data_name is None:
            raise ValueError("Please provide the name of the verify dataset!")

        if self.split == 'train':
            with open(os.path.join(data_folder, self.train_data_name + '_train_images.json'), 'r') as j:
                self.images = json.load(j)
        elif self.split == 'test':
            with open(os.path.join(data_folder, self.test_data_name + '_test_images.json'), 'r') as j:
                self.images = json.load(j)
        else:
            with open(os.path.join(data_folder, self.verify_data_name + '_verify_images.json'), 'r') as j:
                self.images = json.load(j)

        self.transform = ImageTransforms(split=self.split,
                                         hr_img_type=self.hr_img_type)
    
    def __getitem__(self, i):
        img = Image.open(self.images[i], mode='r')
        img = img.convert('RGB')
        hr_img_RGB = self.transform(img)
        hr_img_Bayer = RGB2Bayer(torch.reshape(hr_img_RGB, [3, 128, 128]), 'grbg')
        return hr_img_Bayer, hr_img_RGB

    def __len__(self):
        return len(self.images)
