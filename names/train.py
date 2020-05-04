import torch
import torch.nn as nn
# import numpy as np
# import itertools

from model import NameClassifier, device
from data import train_loader, validation_loader, test_loader  # noqa

cnn = NameClassifier().to(device).double()
weights = [0.10, 1.0]
class_weights = torch.DoubleTensor(weights).to(device)
print('Class weights: ', class_weights)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
# loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), 0.001)


def train(name, category):
    optimizer.zero_grad()
    out = cnn.forward(name)
    # out = out.flatten()
    # print(out.shape, category.shape, out, category)
    loss = loss_fn(out, category.to(device))
    training_loss = loss.item()
    loss.backward()
    optimizer.step()
    return training_loss


def evaluate(samples):
    with torch.no_grad():
        right = 0
        for data in samples:
            name, category = data
            output = cnn.forward(name)
            # output = output.flatten()
            loss = loss_fn(output, category.to(device)).item()
            _, prediction = torch.max(output.data, 1)

            # label = category.numpy().astype(int)
            # output = output.data.cpu().numpy()
            # pred = np.heaviside(output, 0).astype(int)
            correct = torch.sum(prediction == category.data)
            # total = len(category)
            # print(correct, total)
            right += int(correct)
            # error += (total - correct)
        # print(right, len(samples))
        accuracy = (right / len(samples)) * 100.0
        return loss, accuracy


# train_losses = []
cnn.train()
for idx, item in enumerate(train_loader):
    name, category = item
    # print(name.shape, category.shape)
    training_loss = train(name, category)
    # train_losses.append(loss)
    if idx > 0 and idx % 1000 == 0:
        print(f'Index: {idx}, loss: {training_loss}')
    if idx > 0 and idx % 10000 == 0:
        cnn.eval()
        loss, accuracy = evaluate(validation_loader)
        print(f'Training Loss: {training_loss}, Validation Loss: {loss}, Validation Accuracy: {accuracy}%')  # noqa
        cnn.train()
