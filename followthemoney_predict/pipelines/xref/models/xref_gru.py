from itertools import chain
from operator import itemgetter
from collections import Counter
import logging

from normality import normalize

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

from tqdm import tqdm

from .xref_model import XrefModel


class PandasDatasetWraper:
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def __len__(self):
        return len(self.df)


class TextEmbeddModule(nn.Module):
    def __init__(self, vocab_size=8192, embed_dim=512, num_class=2):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc1 = nn.Linear(2 * embed_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_class)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, left_names, right_names, **kwargs):
        embedded1 = self.embedding(left_names)
        embedded2 = self.embedding(right_names)
        X = torch.cat((embedded1, embedded2), axis=1)
        X = self.relu(self.fc1(X))
        X = self.softmax(self.fc2(X))
        return X


class Vocabulary:
    def __init__(self, vocabulary_size, ngram):
        self.vocabulary_size = vocabulary_size
        self.ngram = ngram

    def _getngrams(self, item):
        ngram = self.ngram
        item_clean = normalize(item, latinize=True).lower()
        for i in range(len(item_clean) - ngram):
            yield item_clean[i : i + ngram]

    def fit(self, X, *args, **kwargs):
        ngram = self.ngram
        vocabulary_counts = Counter()
        n_docs = 0
        for item in X:
            n_docs += 1
            for ngram in self._getngrams(item):
                vocabulary_counts[ngram] += 1
        max_count = n_docs * 0.9
        while True:
            item, count = vocabulary_counts.most_common(1)[0]
            if count < max_count:
                break
            vocabulary_counts.pop(item)
        self.vocabulary_ = {
            item: idx + 1
            for idx, (item, count) in zip(
                range(self.vocabulary_size), vocabulary_counts.most_common()
            )
        }
        return self

    def transform(self, X, *args):
        for item in X:
            ngrams = list(self._getngrams(item))
            yield [self.vocabulary_[n] for n in ngrams if n in self.vocabulary_]


class XrefGru(XrefModel):
    version = "0.1"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.meta = {}
        self.meta["init_args"] = {
            "vocabulary": {
                "vocabulary_size": 8192,
                "ngram": 3,
            }
        }
        self.clf = {
            "vocabulary": Vocabulary(**self.meta["init_args"]["vocabulary"]),
            "embedding": TextEmbeddModule(),
        }
        super().__init__()

    def transform_data(self, items):
        vocabulary = self.clf["vocabulary"]
        left_names = list(vocabulary.transform(map(itemgetter("left_name"), items)))
        right_names = list(vocabulary.transform(map(itemgetter("right_name"), items)))
        judgements = map(itemgetter("judgement"), items)

        N_l = max(len(d) for d in left_names)
        N_r = max(len(d) for d in right_names)

        X_l = np.zeros((len(items), N_l), dtype=np.long)
        X_r = np.zeros((len(items), N_r), dtype=np.long)
        for i, (l, r) in enumerate(zip(left_names, right_names)):
            X_l[i, : len(l)] = l
            X_r[i, : len(r)] = r
        X_l = torch.tensor(X_l)
        X_r = torch.tensor(X_r)
        y = torch.tensor(list(judgements), dtype=np.int)
        return {"left_names": X_l, "right_names": X_r, "judgement": y}

    def _dataloader(self, df):
        return DataLoader(
            PandasDatasetWraper(df),
            batch_size=128,
            shuffle=True,
            collate_fn=self.transform_data,
            pin_memory=True,
            drop_last=True,
            num_workers=6,
        )

    def _train(self, df, model, optimizer, criterion):
        data = self._dataloader(df)
        train_loss = 0
        train_acc = 0
        for i, X in enumerate(data):
            X = {k: v.to(self.device) for k, v in X.items()}
            optimizer.zero_grad()
            output = model(**X)
            loss = criterion(output, X["judgement"])
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_acc += (output.argmax(1) == X["judgement"]).sum().item()

        return train_loss / len(df), train_acc / len(df)

    def _test(self, df, model, criterion):
        data = self._dataloader(df)
        loss = 0
        acc = 0
        for X in data:
            with torch.no_grad():
                X = {k: v.to(self.device) for k, v in X.items()}
                output = model(**X)
                loss = criterion(output, X["judgement"])
                loss += loss.item()
                acc += (output.argmax(1) == X["judgement"]).sum().item()

        return loss / len(df), acc / len(df)

    def fit(self, df):
        vocabulary = self.clf["vocabulary"]
        embedding = self.clf["embedding"]

        logging.debug(f"Training GRU model on dataframe with shape: {df.shape}")
        train, test = self.prepair_train_test(df)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        name_iter = chain(train["left_name"], train["right_name"])
        if logging.root.level == logging.DEBUG:
            name_iter = tqdm(
                name_iter, desc="Fitting Count Vectorizer", total=train.shape[0] * 2
            )
        vocabulary.fit(name_iter)

        # X_train = dict(
        # name_left=self.transform_data(train['left_name']),
        # name_right=self.transform_data(train['right_name'])
        # )
        # y_train = np.zeros((len(train), 2))
        # y_train[:, train['judgement'].astype(np.int)] = 1

        criterion = torch.nn.NLLLoss().to(self.device)
        optimizer = torch.optim.SGD(embedding.parameters(), lr=4.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

        embedding = embedding.to(self.device)
        for epoch in range(500):
            train_loss, train_acc = self._train(df, embedding, optimizer, criterion)
            scheduler.step()
            valid_loss, valid_acc = self._test(test, embedding, criterion)

            print(f"Epoch: {epoch + 1}")
            print(
                f"\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)"
            )
            print(
                f"\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)"
            )
