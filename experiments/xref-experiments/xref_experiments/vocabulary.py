from collections import Counter
import orjson as json

from followthemoney import model
from normality import normalize

from xref_experiments import utils


def preprocessor(text):
    return normalize(text, lowercase=True, collapse=True, latinize=True)


def proxy_values(proxy):
    for prop, value in proxy.itervalues():
        if value:
            yield prop.type, prop.name, value


def tokenize(group, prop, value, ngram=3):
    if group.name in {"countries", "language", "topic"}:
        yield value
    else:
        for i in range(max(1, len(value) - ngram + 1)):
            yield value[i : i + ngram]


def stream_vocabulary_record(item, tokenize_args=None):
    tokenize_args = tokenize_args or {}
    if utils.filter_schema(item):
        proxy = model.get_proxy(item)
        for group, prop, value in proxy_values(proxy):
            if group == "entity":
                continue
            value = preprocessor(value)
            yield proxy, group, prop, tokenize(group, prop, value, **tokenize_args)


def stream_vocabulary_records(items, tokenize_args=None):
    tokenize_args = tokenize_args or {}
    for item in items:
        yield from stream_vocabulary_record(item, tokenize_args=tokenize_args)


class Vocabulary:
    def __init__(
        self,
        lookup=None,
        counts=None,
        frozen=True,
        unk="__UNK__",
        metadata=None,
        n_features=None,
    ):
        self.lookup = lookup or dict()
        self.lookup_inv = None
        self.counts = counts or Counter()
        self.metadata = metadata or {}
        self.n_features = n_features
        self.frozen = frozen
        self.unk = unk

    def freeze(self):
        self.lookup = {
            key: i + 2
            for i, (key, _) in enumerate(self.counts.most_common(self.n_features))
        }
        self.lookup[self.unk] = 1
        self.counts = None
        self.frozen = True

    def add(self, key):
        self.counts[key] += 1

    def add_multi(self, keys):
        self.counts.update(keys)

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, key):
        assert self.frozen
        idx = self.lookup.get(key)
        if idx is not None:
            return idx
        return self.lookup[self.unk]

    def invert(self, indicies):
        try:
            indicies = iter(indicies)
        except TypeError:
            indicies = [indicies]
        assert self.frozen
        if self.lookup_inv is None:
            self.lookup_inv = {i: t for t, i in self.lookup.items()}
        return [self.lookup_inv.get(i) for i in indicies]

    def to_list(self):
        values = list(self.lookup.keys())
        values.sort(key=self.lookup.get)
        return values

    @classmethod
    def from_file(cls, fd):
        data = json.loads(fd.read())
        return cls(**data)

    def to_file(self, fd):
        fd.write(
            json.dumps(
                {
                    "lookup": self.lookup,
                    "frozen": self.frozen,
                    "unk": self.unk,
                    "metadata": self.metadata,
                }
            )
        )
