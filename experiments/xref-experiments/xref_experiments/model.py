import torch
from torch import nn

from tqdm import tqdm

from xref_experiments.create_model_data import (
    load_vocabularies,
    ParquetModelData,
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
    def __init__(self, n_vocab, n_embed):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed

        self.embedding = nn.EmbeddingBag(
            n_vocab, n_embed, mode="mean", scale_grad_by_freq=True
        )

    def forward(self, ngrams):
        ngrams_flat = torch.cat(ngrams)
        ngrams_offset = torch.tensor(
            [0] + [n.shape[0] for n in ngrams[:-1]], device=self.device
        ).cumsum(dim=0)
        return self.embedding(ngrams_flat, ngrams_offset)

    def similar(self, ngrams, pool=None, k=10):
        e = self([ngrams])[0]
        if pool is None:
            target = self.embedding.weight
        else:
            target = self(pool)
        cossim = torch.cosine_similarity(e.view(1, -1), target)
        return torch.topk(cossim, k=k, sorted=True)


class PropertySkipgramModel(Module):
    def __init__(self, pmd, **kwargs):
        super().__init__()
        self.pmd = pmd

        kwargs.setdefault("n_vocab", len(pmd.vocabularies["ngrams"]))

        self.property_embedding = PropertyEmbedding(**kwargs)
        self.similarity = torch.nn.CosineSimilarity()

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

        y_pred = self.similarity(X_left, X_right)
        return y_pred

    def loss(self, predict, target, negative_sampling):
        C = 1.0 / negative_sampling - 1.0
        weights = (1 - target) * C + 1
        return (weights * (predict - target).pow(2)).sum() / weights.sum()

    def fit(
        self,
        n_epochs,
        lr,
        weight_decay=0.01,
        max_norm=1,
        negative_sampling=1,
        batch_size=8192,
        n_train_samples=None,
        n_valid_samples=None,
        load_n_groups=10,
        random_seed=None,
    ):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.history_ = {f: {"loss": [], "acc": []} for f in ("train", "valid")}
        for n_epoch in range(n_epochs):
            train_loss_total = 0
            train_acc_total = 0
            self.train()
            print(f"******************* EPOCH {n_epoch} ***********************")
            with tqdm(desc="Training", total=n_train_samples) as pbar:
                data_train = self.pmd.skipgrams(
                    "train",
                    batch_size=batch_size,
                    max_samples=n_train_samples,
                    negative_sampling=negative_sampling,
                    load_n_groups=load_n_groups,
                    random_seed=random_seed,
                )
                for n_batch, X in enumerate(data_train):
                    self.zero_grad()
                    y_true = torch.as_tensor(
                        X["target"].values, dtype=torch.float, device=self.device
                    )
                    y_pred = self(X)
                    loss = self.loss(y_pred, y_true, negative_sampling)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
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
                self.history_["train"]["loss"].append(train_loss)
                self.history_["train"]["acc"].append(train_acc)

            self.eval()
            valid_loss_total = 0
            valid_acc_total = 0
            with tqdm(desc="Validating", total=n_valid_samples) as pbar:
                data_valid = self.pmd.skipgrams(
                    "valid",
                    batch_size=batch_size,
                    max_samples=n_valid_samples,
                    negative_sampling=negative_sampling,
                    load_n_groups=load_n_groups,
                    random_seed=random_seed,
                )
                for n_batch, X in enumerate(data_valid):
                    y_true = torch.as_tensor(
                        X["target"].values, dtype=torch.float, device=self.device
                    )
                    y_pred = self(X)
                    loss = self.loss(y_pred, y_true, negative_sampling)
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
                self.history_["valid"]["loss"].append(valid_loss)
                self.history_["valid"]["acc"].append(valid_acc)
        return self.history_


if __name__ == "__main__":
    vocabularies = load_vocabularies("/scratch/xref-experiments/vocabulary/")
    data_dir = "/scratch/xref-experiments/model-data/"
    with ParquetModelData(data_dir, "r", vocabularies=vocabularies) as data:
        model = PropertySkipgramModel(data, n_embed=2048)
        model = model.cuda()
        history = model.fit(
            n_epochs=1000,
            lr=0.001,
            weight_decay=0.05,
            n_train_samples=16_000,
            n_valid_samples=1_000,
            # n_train_samples=16_000_000,
            # n_valid_samples=1_000_000,
            negative_sampling=5,
            load_n_groups=2,
            random_seed=42,
        )
