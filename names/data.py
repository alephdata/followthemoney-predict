import os
import csv
import random
# import fingerprints
from collections import defaultdict
from torch.utils.data import IterableDataset, DataLoader

from model import NameClassifier
from util import normalize

# random.seed(23)
file_path = os.path.abspath(__file__)
data_path = os.path.join(os.path.dirname(file_path),
                         '../work/names.filtered.shuffled.csv')
TOTAL = 1000000
OFFSET = 0  # int(random.randint(0, TOTAL))
TEST_SIZE = int(TOTAL * 0.05)
VALIDATION_SIZE = int(TOTAL * 0.05)


def iter_items():
    classes = defaultdict(int)
    with open(data_path, 'r') as fh:
        for row in csv.reader(fh):
            category, _, name = row
            total = max(1, sum(classes.values()))
            # print(category, (classes[category] / total), total)
            if (classes[category] / total) > 0.51:
                # print(category, 'skip')
                continue
            name = normalize(name)
            if name is not None and len(name) > 1:
                # print(name, category)
                yield name, category
            # if category == 'Company':
            #     removed = fingerprints.remove_types(name, normalize)
            #     if removed is not None and removed != name:
            #         # print(name, removed, category)
            #         yield removed, category
            classes[category] += 1


class NameDataset(IterableDataset):

    def __init__(self, names, names_len=None):
        self.names = names
        self.names_len = names_len or len(names)

    def __iter__(self):
        return map(self.encode, self.names)

    def __len__(self):
        return self.names_len

    def classes(self):
        counts = defaultdict(int)
        for (_, category) in self.names:
            counts[category] += 1
        return dict(counts)

    def encode(self, item):
        name, category = item
        name = NameClassifier.encode_name(name)
        label = NameClassifier.encode_category(category)
        return name, label


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
REMAINING = TOTAL - TEST_SIZE - VALIDATION_SIZE
train_dataset = NameDataset(items, REMAINING)

print('Training: ', len(train_dataset))
print('Validation: ', len(validation_dataset), ' / classes: ', repr(validation_dataset.classes()))  # noqa
print('Test: ', len(test_dataset), ' / classes: ', repr(test_dataset.classes()))  # noqa

train_loader = DataLoader(train_dataset, batch_size=8)
validation_loader = DataLoader(validation_dataset, batch_size=1000)
test_loader = DataLoader(test_dataset, batch_size=1000)
