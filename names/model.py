import torch
import torch.nn as nn
import numpy as np


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


class NameClassifier(nn.Module):
    ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789 \'.?'
    WIDTH = len(ALPHABET)
    LENGTH = 250
    CATEGORIES = ['Person', 'Company']

    def __init__(self):
        super(NameClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(self.LENGTH, 32, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.ReLU()
        )
        self.drop_out = nn.Dropout()
        # self.lin = nn.Linear(512, len(self.CATEGORIES))
        self.lin = nn.Linear(512, len(self.CATEGORIES))
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, name):
        name = name.to(device)
        name = self.conv(name)
        name = name.view(name.shape[0], -1)
        name = self.lin(name)
        name = self.out(name)
        # print(name.shape)
        return name

    @classmethod
    def encode_name(cls, text):
        text = text[:cls.LENGTH].rjust(cls.LENGTH, '?')
        indexes = [cls.ALPHABET.find(c) for c in text]
        return np.eye(cls.WIDTH)[indexes]

    @classmethod
    def encode_category(cls, category):
        return cls.CATEGORIES.index(category)

    @classmethod
    def category_name(cls, category):
        return cls.CATEGORIES[category]
