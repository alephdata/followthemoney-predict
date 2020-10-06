"""
NOTE: try using `ignite` to simplify pytorch transformations?
"""
import logging
from itertools import chain, cycle

import torch
import torch.nn as nn
from followthemoney.exc import InvalidData
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

from followthemoney_predict.lib.vocabulary import Vocabulary

from .xref_torch_model import XrefTorchModel


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


class XrefGru(XrefTorchModel):
    version = "0.1"

    def __init__(self, batch_size=512):
        self.__device = None

        self.meta = {"batch_size": batch_size}
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

    def transform_data(self, samples):
        N = len(samples)
        N_l = max(len(s["left_name"]) for s in samples)
        N_r = max(len(s["right_name"]) for s in samples)

        y = torch.zeros(N, dtype=torch.long)
        weight = torch.zeros(N, dtype=torch.float)
        X_l = torch.zeros((N, N_l), dtype=torch.long)
        X_r = torch.zeros((N, N_r), dtype=torch.long)
        left_seq_len = torch.zeros(N, dtype=torch.long)
        right_seq_len = torch.zeros(N, dtype=torch.long)
        for i, sample in enumerate(samples):
            left_name = sample["left_name"]
            right_name = sample["right_name"]
            X_l[i, : len(left_name)] = torch.tensor(left_name)
            X_r[i, : len(right_name)] = torch.tensor(right_name)
            left_seq_len[i] = len(left_name)
            right_seq_len[i] = len(right_name)
            y[i] = sample["judgement"]
            weight[i] = sample["weight"]
        return {
            "left_names": X_l,
            "right_names": X_r,
            "left_seq_len": left_seq_len,
            "right_seq_len": right_seq_len,
            "judgement": y,
            "weight": weight,
        }

    def create_dataloader_entities(self, ftm_model, entity_pairs, shuffle=True):
        vocabulary = self.clf["vocabulary"]
        samples = []
        filtered_idxs = []
        sample_group_len = []
        for i, (A, B) in enumerate(entity_pairs):
            try:
                ftm_model.common_schema(A.schema, B.schema)
            except InvalidData:
                filtered_idxs.append(i)
                continue
            A_names = list(filter(None, map(vocabulary, set(A.names))))
            B_names = list(filter(None, map(vocabulary, set(B.names))))
            if not len(A_names) or not len(B_names):
                filtered_idxs.append(i)
            elif len(A_names) > 1 or len(B_names) > 1:
                sample_group_len.append((len(samples), len(A_names) * len(B_names)))
            for A_name in A_names:
                for B_name in B_names:
                    samples.append(
                        {
                            "left_name": A_name,
                            "right_name": B_name,
                            "weight": 1,
                            "judgement": -1,
                        }
                    )
        return self.create_dataloader_samples(
            samples,
            shuffle,
            filtered_idxs=filtered_idxs,
            sample_group_len=sample_group_len,
        )

    def create_dataloader_dataframe(self, df, shuffle=True):
        vocabulary = self.clf["vocabulary"]
        left_names = vocabulary.transform(df.left_name)
        right_names = vocabulary.transform(df.right_name)
        samples = []
        filtered_idxs = []
        if hasattr(df, "weight"):
            weights = df.weight
        else:
            weights = cycle([1])
        _iter = tqdm(
            zip(left_names, right_names, weights, df.judgement),
            desc="Pre-processing Data",
            total=df.shape[0],
        )
        for i, (left_name, right_name, weight, judgement) in enumerate(_iter):
            if not len(left_name) or not len(right_name):
                filtered_idxs.append(i)
            else:
                samples.append(
                    {
                        "left_name": left_name,
                        "right_name": right_name,
                        "weight": weight,
                        "judgement": judgement,
                    }
                )
        return self.create_dataloader_samples(
            samples, shuffle, filtered_idxs=filtered_idxs
        )

    def fit(self, df):
        vocabulary = self.clf["vocabulary"]
        embedding = self.clf["embedding"].to(self.device)

        logging.info(f"Training GRU model on dataframe with shape: {df.shape}")
        train, test = self.prepair_train_test(df, weight_class=False)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        name_iter = chain(train["left_name"], train["right_name"])
        if logging.root.level == logging.DEBUG:
            name_iter = tqdm(
                name_iter, desc="Fitting Count Vectorizer", total=train.shape[0] * 2
            )
        vocabulary.fit(name_iter)

        N = df.shape[0]
        n_positive = df.judgement.sum()
        n_per_class = [N - n_positive, n_positive]
        class_weights = torch.FloatTensor([1 - (x / N) for x in n_per_class])

        criterion_unweighted = torch.nn.CrossEntropyLoss(
            weight=class_weights, reduction="none"
        ).to(self.device)
        criterion = self.sample_weighted_criterion(criterion_unweighted)
        optimizer = torch.optim.SGD(embedding.parameters(), lr=0.05, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.9)

        train_data = self.create_dataloader_dataframe(train)
        test_data = self.create_dataloader_dataframe(test, shuffle=False)

        model, history = self.fit_torch(
            embedding,
            train_data,
            test_data,
            criterion,
            optimizer,
            scheduler,
        )
        self.clf["embedding"] = model
        self.meta["history"] = history

        scores = self.describe(test)
        self.meta["scores"] = scores
        return self

    def predict(self, df):
        embedding = self.clf["embedding"]
        dataloader = self.create_dataloader_dataframe(df, shuffle=False)
        return self.predict_torch(embedding, dataloader)

    def compare(self, ftm_model, A, B):
        embedding = self.clf["embedding"]
        dataloader = self.create_dataloader_entities(ftm_model, [(A, B)], shuffle=False)
        return self.predict_torch(embedding, dataloader)[0][1]

    def compare_batch(self, ftm_model, A, Bs, reduction=None):
        embedding = self.clf["embedding"]
        pairs = ((A, B) for B in Bs)
        dataloader = self.create_dataloader_entities(ftm_model, pairs, shuffle=False)
        return self.predict_torch(embedding, dataloader, reduction=reduction)
