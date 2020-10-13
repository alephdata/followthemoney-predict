from collections import Counter

from normality import normalize


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
            yield self(item)

    def __call__(self, item):
        ngrams = list(self._getngrams(item))
        return [self.vocabulary_[n] for n in ngrams if n in self.vocabulary_]
