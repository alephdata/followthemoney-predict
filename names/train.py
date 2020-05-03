import torch
import torch.nn as nn
import numpy as np
# import itertools

from model import NameClassifier, device
from data import train_loader, validation_loader, test_loader  # noqa

cnn = NameClassifier().to(device).double()
loss_fn = nn.NLLLoss()
# print(list(cnn.parameters()))
optimizer = torch.optim.Adam(cnn.parameters(), 0.001)


def train(name, category):
    optimizer.zero_grad()
    out = cnn.forward(name)
    # out = out.flatten()
    # print(out.shape, category.shape)
    loss = loss_fn(out, category.to(device))
    training_loss = loss.item()
    loss.backward()
    optimizer.step()
    return training_loss


def evaluate(samples):
    cnn.eval()
    with torch.no_grad():
        right, error = 0, 0
        for data in samples:
            name, category = data
            output = cnn.forward(name)
            # output = output.flatten()
            loss = loss_fn(output, category.to(device)).item()

            label = category.numpy().astype(int)
            output = output.data.cpu().numpy()
            pred = np.heaviside(output, 0).astype(int)
            correct = np.sum(label == pred)
            total = len(label)
            right += correct
            error += (total - correct)
    accuracy = (right/len(samples)) * 100.0
    return loss, accuracy


# train_losses = []
for idx, item in enumerate(train_loader):
    name, category = item
    # print(name.shape, category.shape)
    training_loss = train(name, category)
    # train_losses.append(loss)
    if idx > 0 and idx % 1000 == 0:
        print(f'Index: {idx}, loss: {training_loss}')
    if idx > 0 and idx % 10000 == 0:
        loss, accuracy = evaluate(validation_loader)
        print(f'Training Loss: {training_loss}, Validation Loss: {loss}, Validation Accuracy: {accuracy}%')  # noqa


# def test():
#     net.eval()
#     loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
#     with torch.no_grad():
#         right, error = 0, 0
#         for data in test_loader:
#             ent1, ent2, label = data
#             label = label.double()
#             output = net.forward(ent1, ent2)
#             output = output.flatten()
#             validation_loss = loss_fn(output, label.to(device)).item()

#             label = label.numpy().astype(int)
#             output = output.data.cpu().numpy()
#             pred = np.heaviside(output, 0).astype(int)
#             correct = np.sum(label == pred)
#             total = len(label)
#             right += correct
#             error += (total - correct)
#     print(f'Test Loss:{validation_loss}, Test Accuracy: {right/len(test_dataset)*100}%')
# test()