import torch
from torch import nn

from xref_experiments.utils import last_items


def packed_elementwise_apply(fn, packed_sequence):
    """
    applies a pointwise function fn to each element in packed_sequence
    from: https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184/4
    """
    return nn.utils.rnn.PackedSequence(
        fn(packed_sequence.data), packed_sequence.batch_sizes
    )


class PropertyEmbedding(nn.Module):
    def __init__(self, n_vocab, n_embed, n_gru_hidden, n_gru_layers=2, dropout=0.5):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.__dummy = nn.Parameter(torch.empty(0))

        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.recurrent = nn.GRU(
            n_embed,
            hidden_size=n_gru_hidden,
            num_layers=n_gru_layers,
            dropout=dropout,
            bidirectional=True,
        )

    @property
    def device(self):
        return self.__dummy.device

    def forward(self, ngrams):
        ngrams_packed = nn.utils.rnn.pack_sequence(ngrams, enforce_sorted=False).to(
            self.device
        )
        embed_packed = packed_elementwise_apply(self.embedding, ngrams_packed)
        X_packed, h_packed = self.recurrent(embed_packed)
        X = last_items(X_packed, unsort=True)
        return X


class PropertySkipgramModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pe_left = PropertyEmbedding(*args, **kwargs)
        self.pe_right = PropertyEmbedding(*args, **kwargs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        ngrams_left = X["ngrams_x"].map(torch.from_numpy).to_list()
        ngrams_right = X["ngrams_y"].map(torch.from_numpy).to_list()

        X_left = self.pe_left(ngrams_left)
        X_right = self.pe_right(ngrams_right)

        y_pred = self.sigmoid((X_left * X_right).sum(-1))
