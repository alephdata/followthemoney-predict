import torch
import torch.nn as nn

from model import NameClassifier, device

cnn = NameClassifier().to(device).double()
cnn.load_state_dict(torch.load('names.pyts'))
cnn.eval()


def classify(text):
    data = NameClassifier.encode_name(text)
    data = torch.tensor(data)
    out = cnn.forward(data)
    _, prediction = torch.max(out.data, 1)
    print(prediction)


classify("Friedrich Lindenberg")

