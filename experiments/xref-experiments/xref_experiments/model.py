from itertools import islice

import torch
from torch import nn
from torch import optim

from tqdm import tqdm

from xref_experiments.utils import last_items, sorted_last_indices, sorted_first_indices
from xref_experiments.create_model_data import (
    load_vocabularies,
    ParquetModelData,
    multiprocess_skipgram,
)


def packed_elementwise_apply(fn, packed_sequence):
    """
    applies a pointwise function fn to each element in packed_sequence
    from: https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184/4
    """
    return nn.utils.rnn.PackedSequence(
        fn(packed_sequence.data), packed_sequence.batch_sizes
    )


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return self._dummy.device


class PropertyEmbedding(Module):
    def __init__(self, n_vocab, n_embed, n_gru_hidden, n_gru_layers=2, dropout=0.5):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.n_gru_hidden = n_gru_hidden
        self.n_gru_layers = n_gru_layers
        self.dropout = dropout

        self.embedding = nn.EmbeddingBag(n_vocab, n_embed, mode="sum")
        # self.recurrent = nn.GRU(
        # n_embed,
        # hidden_size=n_gru_hidden,
        # num_layers=n_gru_layers,
        # dropout=dropout,
        # bidirectional=True,
        # )

    def forward(self, ngrams):
        ngrams_flat = torch.cat(ngrams)
        ngrams_offset = torch.tensor(
            [0] + [n.shape[0] for n in ngrams[:-1]], device=self.device
        ).cumsum(dim=0)
        return self.embedding(ngrams_flat, ngrams_offset)

        # ngrams_packed = nn.utils.rnn.pack_sequence(ngrams, enforce_sorted=False).to(
        #     self.device
        # )
        # embed_packed = packed_elementwise_apply(self.embedding, ngrams_packed)
        # X_packed, h_packed = self.recurrent(embed_packed)
        # X = last_items(X_packed, unsort=True)
        # return X


class PropertySkipgramModel(Module):
    def __init__(self, pmd, **kwargs):
        super().__init__()
        self.pmd = pmd

        kwargs.setdefault("n_vocab", len(pmd.vocabularies["ngrams"]))

        self.property_embedding = PropertyEmbedding(**kwargs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        ngrams_left = (
            X["ngrams_x"]
            .map(lambda n: torch.as_tensor(n, device=self.device))
            .to_list()
        )
        ngrams_right = (
            X["ngrams_y"]
            .map(lambda n: torch.as_tensor(n, device=self.device))
            .to_list()
        )

        X_left = self.property_embedding(ngrams_left)
        X_right = self.property_embedding(ngrams_right)

        y_pred = self.sigmoid((X_left * X_right).sum(-1))
        return y_pred

    def fit(
        self, lr, n_epochs, batch_size=8192, n_train_samples=None, n_valid_samples=None
    ):
        loss_fn = nn.BCELoss().to(self.device)
        optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        history = {f: {"loss": [], "acc": []} for f in ("train", "valid")}
        for n_epoch in range(n_epochs):
            train_loss_total = 0
            train_acc_total = 0
            self.train()
            print(f"******************* EPOCH {n_epoch} ***********************")
            with tqdm(desc="Training", total=n_train_samples) as pbar:
                data_train = self.pmd.skipgrams(
                    "train", batch_size=batch_size, max_samples=n_train_samples
                )
                for n_batch, X in enumerate(data_train):
                    y_true = torch.as_tensor(
                        X["target"].values, dtype=torch.float, device=self.device
                    )
                    y_pred = self(X)
                    loss = loss_fn(y_pred, y_true)
                    loss.backward()
                    optimizer.step()
                    train_loss_total += loss.item()
                    train_acc_total += (
                        (y_pred > 0.5) == y_true
                    ).sum().item() / X.shape[0]
                    pbar.update(X.shape[0])

                    if (n_batch + 1) % 50 == 0:
                        train_loss = train_loss_total / (n_batch + 1)
                        train_acc = train_acc_total / (n_batch + 1)
                        pbar.set_description(
                            f"Training [loss:{train_loss:0.4f}][acc:{train_acc:0.4f}]"
                        )
                train_loss = train_loss_total / (n_batch + 1)
                train_acc = train_acc_total / (n_batch + 1)
                pbar.write(
                    f"TRAIN FINAL epoch: {n_epoch}, n_batch: {n_batch}, loss: {train_loss}, acc: {train_acc}"
                )
                history["train"]["loss"].append(train_loss)
                history["train"]["acc"].append(train_acc)

            self.eval()
            valid_loss_total = 0
            valid_acc_total = 0
            with tqdm(desc="Validating", total=n_valid_samples) as pbar:
                data_valid = self.pmd.skipgrams(
                    "valid", batch_size=batch_size, max_samples=n_valid_samples
                )
                for n_batch, X in enumerate(data_valid):
                    y_true = torch.as_tensor(
                        X["target"].values, dtype=torch.float, device=self.device
                    )
                    y_pred = self(X)
                    loss = loss_fn(y_pred, y_true)
                    valid_loss_total += loss.item()
                    valid_acc_total += (
                        (y_pred > 0.5) == y_true
                    ).sum().item() / X.shape[0]
                    pbar.update(X.shape[0])
                valid_loss = valid_loss_total / (n_batch + 1)
                valid_acc = valid_acc_total / (n_batch + 1)
                pbar.write(
                    f"VALID: n_batch: {n_batch}, loss: {valid_loss}, acc: {valid_acc}"
                )
                history["valid"]["loss"].append(valid_loss)
                history["valid"]["acc"].append(valid_acc)
        return history


if __name__ == "__main__":
    vocabularies = load_vocabularies("/scratch/xref-experiments/vocabulary/")
    data_dir = "/scratch/xref-experiments/model-data/"
    with ParquetModelData(data_dir, "r", vocabularies=vocabularies) as data:
        model = PropertySkipgramModel(
            data, n_embed=512, n_gru_hidden=64, n_gru_layers=2
        )
        # model = model.cuda()
        history = model.fit(
            lr=1e-3, n_epochs=100, n_train_samples=5_000_000, n_valid_samples=100_000
        )
