import sys
from pathlib import Path
import random

from tqdm import tqdm
import orjson as json

from xref_experiments import vocabulary as V
from xref_experiments.data import entity_generator


def create_vocabulary(basedir, items, ngram=3):
    vocabs = {}
    vocabs["ngrams"] = V.Vocabulary(n_features=200_000, metadata={"ngram": ngram})
    vocabs["group"] = V.Vocabulary()
    vocabs["property"] = V.Vocabulary()
    vocabs["schema"] = V.Vocabulary()

    for proxy, group, prop, tokens in V.stream_vocabulary_records(
        items, tokenize_args={"ngram": ngram}
    ):
        vocabs["schema"].add(str(proxy.schema.name))
        vocabs["group"].add(str(group))
        vocabs["property"].add(str(prop))
        vocabs["ngrams"].add_multi(tokens)

    for name, vocab in vocabs.items():
        print("Saving vocab:", name)
        vocab.freeze()
        with open(basedir / f"{name}.json", "wb+") as fd:
            vocab.to_file(fd)
    return vocabs


def stdin_reader():
    with tqdm("STDIN", unit_divisor=1024, unit="B", unit_scale=True) as pbar:
        for line in tqdm(sys.stdin):
            if random.random() < 0.25:
                yield json.loads(line)
            pbar.update(len(line))


if __name__ == "__main__":
    basedir = Path("/data/xref-experiments/vocabulary/")
    basedir.mkdir(exist_ok=True, parents=True)

    data_path = Path("/data/xref-experiments/occrp-data-exports/data/")
    data_exports = list(data_path.glob("*/*.json"))
    items = entity_generator(
        data_exports, entities_per_file=500_000, resevour_sample=True
    )
    # items = stdin_reader()

    fname = create_vocabulary(basedir, items)
    print("Created vocabulary:", fname)
