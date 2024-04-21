from torch.utils.data import dataset
import csv
import numpy as np
import torch

class LoadData(dataset.Dataset):

    def __init__(self, train):
        super(LoadData, self).__init__()
        if train:
            with open('mnist_train.csv','r') as f:
                self.data = list(csv.reader(f))
        else:
            with open('mnist_test.csv','r') as f:
                self.data = list(csv.reader(f))
        for i, item in enumerate(self.data):
            self.data[i] = [float(x) for x in item]

    def __getitem__(self, index):
        item = self.data[index]
        # Convert to tensors and scale image data to [0, 1]
        img = torch.tensor(item[1:], dtype=torch.float32).view(28, 28) / 255.0
        label = torch.tensor(item[0], dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.data)