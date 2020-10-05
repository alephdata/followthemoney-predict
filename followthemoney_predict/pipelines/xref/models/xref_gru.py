"""
NOTE: try using `ignite` to simplify pytorch transformations?
"""
from itertools import chain, count
from collections import Counter, defaultdict
import logging
import copy

from normality import normalize

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from tqdm import tqdm

from .xref_model import XrefModel


class TextEmbeddModule(nn.Module):
    def __init__(
        self, vocab_size=8192, embed_dim=64, hidden_size=64, num_class=2, dropout=0.5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers=1, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_class, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward_branch(self, names, lengths):
        embed = self.embedding(names)
        packed = pack_padded_sequence(
            embed, lengths, enforce_sorted=False, batch_first=True
        )
        _, h = self.gru(packed)
        h = h.view(names.shape[0], -1)
        return self.dropout(h)

    def forward(self, left_names, right_names, left_seq_len, right_seq_len, **kwargs):
        embed_left = self.forward_branch(left_names, left_seq_len)
        embed_right = self.forward_branch(right_names, right_seq_len)
        X = torch.cat((embed_left, embed_right), axis=1)
        X = self.softmax(self.fc(X))
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
                range(self.vocabulary_size - 1), vocabulary_counts.most_common()
            )
        }
        self.vocabulary_[None] = 0
        return self

    def transform(self, X, *args):
        for item in X:
            ngrams = list(self._getngrams(item))
            yield [self.vocabulary_[n] for n in ngrams if n in self.vocabulary_]


class XrefGru(XrefModel):
    version = "0.1"

    def __init__(self, batch_size=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size
        self.meta = {}
        self.meta["init_args"] = {
            "vocabulary": {
                "vocabulary_size": 8192,
                "ngram": 3,
            }
        }
        self.clf = {
            "vocabulary": Vocabulary(**self.meta["init_args"]["vocabulary"]),
            "embedding": TextEmbeddModule().to(self.device),
        }
        super().__init__()

    def transform_data(self, items):
        N = len(items)
        N_l = max(len(d[0]) for d in items)
        N_r = max(len(d[1]) for d in items)

        y = torch.zeros(N, dtype=torch.long)
        X_l = torch.zeros((N, N_l), dtype=torch.long)
        X_r = torch.zeros((N, N_r), dtype=torch.long)
        left_seq_len = torch.zeros(N, dtype=torch.long)
        right_seq_len = torch.zeros(N, dtype=torch.long)
        for i, (l, r, j) in enumerate(items):
            X_l[i, : len(l)] = torch.tensor(l)
            X_r[i, : len(r)] = torch.tensor(r)
            left_seq_len[i] = len(l)
            right_seq_len[i] = len(r)
            y[i] = j
        return {
            "left_names": X_l,
            "right_names": X_r,
            "left_seq_len": left_seq_len,
            "right_seq_len": right_seq_len,
            "judgement": y,
        }

    def create_dataloader(self, df, shuffle=True):
        vocabulary = self.clf["vocabulary"]
        left_names = tuple(vocabulary.transform(df.left_name))
        right_names = tuple(vocabulary.transform(df.right_name))
        data = tuple(
            filter(
                lambda l_r_j: len(l_r_j[0]) and len(l_r_j[1]),
                tqdm(
                    zip(left_names, right_names, df.judgement),
                    desc="Preprocessing Data",
                    total=df.shape[0],
                ),
            )
        )
        data_loader = DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.transform_data,
            pin_memory=True,
            drop_last=False,
            num_workers=6,
        )
        data_loader.filtered_idxs = [
            i
            for i in range(df.shape[0])
            if not (len(left_names[i]) and len(right_names[i]))
        ]
        return data_loader

    def _train(self, data, model, optimizer, criterion):
        train_loss = 0
        train_acc = 0
        N = 0
        for i, X in enumerate(
            tqdm(data, desc="Training", leave=True, unit_scale=self.batch_size)
        ):
            N += X["judgement"].shape[0]
            X = {k: v.to(self.device) for k, v in X.items()}
            optimizer.zero_grad()
            output = model(**X)
            loss = criterion(output, X["judgement"])
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_acc += (output.argmax(1) == X["judgement"]).sum().item()

        return train_loss / N, train_acc / N

    def _test(self, data, model, criterion):
        loss = 0
        acc = 0
        N = 0
        N_positive = 0
        outputs = []
        for X in tqdm(data, desc="Testing", leave=True, unit_scale=self.batch_size):
            with torch.no_grad():
                N += X["judgement"].shape[0]
                X = {k: v.to(self.device) for k, v in X.items()}
                output = model(**X)
                N_positive += output[:, -1].sum().item()
                # N_positive += output.sum().item()
                outputs.append(output.cpu())
                loss = criterion(output, X["judgement"])
                loss += loss.item()
                acc += (output.argmax(1) == X["judgement"]).sum().item()
        logging.debug(f"n_positive_pct: {N_positive / N}")
        return loss / N, acc / N

    def fit(self, df):
        vocabulary = self.clf["vocabulary"]
        embedding = self.clf["embedding"]

        logging.info(f"Training GRU model on dataframe with shape: {df.shape}")
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

        N = df.shape[0]
        n_positive = df.judgement.sum()
        n_per_class = [N - n_positive, n_positive]
        class_weights = torch.FloatTensor([1 - (x / N) for x in n_per_class])

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(self.device)
        optimizer = torch.optim.SGD(embedding.parameters(), lr=0.05, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.9)

        train_data = self.create_dataloader(train)
        test_data = self.create_dataloader(test, shuffle=False)

        best_model = None
        best_loss = None
        patience = cur_patience = 15
        history = defaultdict(list)
        for epoch in count(0):
            train_loss, train_acc = self._train(
                train_data, embedding, optimizer, criterion
            )
            scheduler.step()
            valid_loss, valid_acc = self._test(test_data, embedding, criterion)

            logging.debug(f"Epoch: {epoch + 1}")
            logging.debug(
                f"\tLoss: {train_loss:.4e}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)"
            )
            logging.debug(
                f"\tLoss: {valid_loss:.4e}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)"
            )

            history["valid_loss"].append(valid_loss)
            history["valid_acc"].append(valid_acc)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if best_loss is None or valid_loss < best_loss:
                logging.debug(f"New best model: {valid_loss} < {best_loss}")
                best_loss = valid_loss
                best_model = copy.copy(embedding.state_dict())
                cur_patience = patience
            else:
                cur_patience -= 1
                logging.debug(f"Current Patience: {cur_patience}")
            if cur_patience == 0:
                logging.info("Out of patience.")
                break
        self.clf["embedding"].load_state_dict(best_model)
        self.history = history
        return embedding, history

    def predict(self, df):
        dataloader = self.create_dataloader(df, shuffle=False)
        result = np.zeros((df.shape[0] - len(dataloader.filtered_idxs), 2))
        with torch.no_grad():
            for i, X in enumerate(dataloader):
                X = {k: v.to(self.device) for k, v in X.items()}
                output = self.clf["embedding"](**X).cpu()
                N = output.shape[0]
                result[i * N : (i + 1) * N] = output
        skipped_idxs = dataloader.filtered_idxs
        idxs = [si - i for i, si in enumerate(skipped_idxs)]
        return np.insert(result, idxs, [0.5, 0.5], axis=0)
