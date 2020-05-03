import os
import csv
import random
from torch.utils.data import IterableDataset, DataLoader

from model import NameClassifier
from util import normalize

# random.seed(23)
file_path = os.path.abspath(__file__)
data_path = os.path.join(os.path.dirname(file_path),
                         '../work/names.filtered.shuffled.csv')
TOTAL = 1000000
OFFSET = int(random.randint(0, TOTAL * 0.1))
TEST_SIZE = int(TOTAL * 0.05)
VALIDATION_SIZE = int(TOTAL * 0.05)


def iter_items():
    with open(data_path, 'r') as fh:
        for row in csv.reader(fh):
            category, _, name = row
            name = normalize(name)
            if name is not None and len(name) > 1:
                # print(name, category)
                yield name, category


class NameDataset(IterableDataset):

    def __init__(self, names, names_len=None):
        self.names = names
        self.names_len = names_len or len(names)

    def __iter__(self):
        return map(self.encode, self.names)

    def __len__(self):
        return self.names_len

    def encode(self, name):
        text, category = name
        label = NameClassifier.encode_category(category)
        return NameClassifier.encode_name(text), label


def take_fixed(iterable, count):
    done = 0
    items = []
    for item in iterable:
        if done == count:
            break
        done += 1
        items.append(item)
    random.shuffle(items)
    return items


items = iter_items()
take_fixed(items, OFFSET)
validation_dataset = NameDataset(take_fixed(items, VALIDATION_SIZE))
test_dataset = NameDataset(take_fixed(items, TEST_SIZE))
REMAINING = TOTAL - OFFSET - TEST_SIZE - VALIDATION_SIZE
train_dataset = NameDataset(items, REMAINING)

print('Training: ', len(train_dataset))
print('Validation: ', len(validation_dataset))
print('Test: ', len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=16)
validation_loader = DataLoader(validation_dataset, batch_size=1000)
test_loader = DataLoader(test_dataset, batch_size=1000)
