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
    ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789 -\'.?'
    WIDTH = len(ALPHABET)
    LENGTH = 128
    CATEGORIES = ['Person', 'Company']

    def __init__(self):
        super(NameClassifier, self).__init__()
        base = int(self.LENGTH / 2)
        self.conv = nn.Sequential(
            nn.Conv1d(self.LENGTH, base, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(base, int(base / 2), 3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Dropout(),
            nn.Linear(int(base * 4), len(self.CATEGORIES)),
            nn.ReLU()
        )

    def forward(self, name):
        name = name.to(device)
        name = self.conv(name)
        name = name.view(name.shape[0], -1)
        # name = self.lin(name)
        name = self.out(name)
        # print(name.shape)
        return name

    @classmethod
    def encode_name(cls, text):
        text = text[:cls.LENGTH].rjust(cls.LENGTH, '?')
        # print(text)
        indexes = [cls.ALPHABET.find(c) for c in text]
        return np.eye(cls.WIDTH)[indexes]

    @classmethod
    def encode_category(cls, category):
        # if category == 'Person':
        #     return np.array([1, 0])
        # return np.array([0, 1])
        return cls.CATEGORIES.index(category)

    # @classmethod
    # def category_name(cls, category):
    #     return cls.CATEGORIES[category]
