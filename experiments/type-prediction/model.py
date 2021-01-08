#!/usr/bin/env python
from pathlib import Path
import random
from collections import defaultdict
from itertools import chain, islice
import os
import json
import warnings

from tqdm import tqdm
from normality import normalize
import fasttext

try:
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    ANALYSIS_MODULES = True
except ImportError:
    ANALYSIS_MODULES = False

from alephclient.api import AlephAPI
from followthemoney import model
from followthemoney.exc import InvalidData
from ftmstore import Dataset


DATADIR = Path("./data.secret/")
CACHEDIR = Path("/tmp/ftm-type-predict/")
STORE_URI = os.environ["FTM_STORE_URI"]


def clean_value(name):
    return normalize(name, latinize=True, lowercase=True)


def _stream_collection(collection, N=50_000):
    fid = collection["foreign_id"]
    collection_id = collection["id"]
    cachefile = CACHEDIR / f"{fid}.json"
    if not cachefile.exists():
        return
        cachefile.parent.mkdir(parents=True, exist_ok=True)
        cachefile_back = CACHEDIR / "tmp.json"
        dataset = Dataset(f"collection_{collection_id}", origin="aleph")
        with open(cachefile_back, "w+") as fd:
            for entity in islice(dataset.iterate(skip_errors=True), N):
                yield entity
                fd.write(json.dumps(entity.to_dict()))
                fd.write("\n")
        cachefile_back.rename(cachefile)
    else:
        with open(cachefile) as fd:
            for line in fd:
                yield json.loads(line)


def stream_entities():
    api = AlephAPI()
    collections = api.filter_collections("*")
    # collections = filter(lambda c: not c.get("secret", False), collections)
    collections = list(collections)
    random.shuffle(collections)
    for c in tqdm(collections, desc="Collections"):
        cid = c["id"]
        for entity in _stream_collection(c):
            if entity:
                yield cid, entity


def type_datagen(proxy):
    for name, values in proxy.get_type_inverted().items():
        if name in {"entities", "topics", "urls", "languages", "countries"}:
            continue
        if name == "names":
            N = 100_000
        else:
            N = 75_000
        yield name, "types", N, values


def schema_datagen(proxy):
    if proxy.get("name", quiet=True):
        yield proxy.schema.name, "schemas", 100_000, proxy.get("name")


def type_datagen_page(proxy):
    if proxy.schema.is_a(model.get("Page")):
        tokens = chain.from_iterable(
            [b.split(" ") for b in proxy.get("bodyText", quiet=True) if b]
        )
        tokens = list(filter(None, map(str.strip, tokens)))
        ngrams = 1
        while random.random() < 0.4:
            ngrams += 1
        n_samples = 1
        while random.random() < 0.6:
            n_samples += 1
        if len(tokens) < ngrams * n_samples:
            return
        for _ in range(n_samples):
            i = random.randint(0, len(tokens) - ngrams)
            sample = tokens[i : i + ngrams]
            if any(len(t) > 1 for t in sample) and all(len(t) < 24 for t in sample):
                yield "trash", "types", 100_000, [" ".join(tokens[i : i + ngrams])]


def datagen(proxy):
    if proxy.schema.is_a(model.get("Page")):
        yield from type_datagen_page(proxy)
        return
    elif proxy.schema.is_a(model.get("Document")):
        return
    yield from type_datagen(proxy)
    yield from schema_datagen(proxy)


class SamplesFile:
    def __init__(self, fname, category, type, limit):
        self.fname = fname
        self.category = category
        self.type = type
        self.limit = limit
        self._values = []
        self._N = 0

    def full(self):
        return self.limit <= len(self._values)

    def add(self, value, collection_id=None):
        if not value:
            return False
        self._N += 1
        if len(self) >= self.limit:
            i = random.randint(0, self._N)
            if i < len(self):
                self._values[i] = (collection_id, value)
            return False
        else:
            self._values.append((collection_id, value))
            return True

    def close(self):
        name = f"__label__{self.type}"
        self.fname.parent.mkdir(parents=True, exist_ok=True)
        self._values.sort()
        with open(self.fname, "w+") as fd:
            for collection_id, value in self._values:
                fd.write(f"{name} {value}\n")

    def __len__(self):
        return len(self._values)

    def __str__(self):
        pct = len(self) / self.limit * 100
        return f"{self.type}:{pct:0.0f}%"


def download_data(patience=500_000):
    skip_schemas = [model.get("Event"), model.get("Mention")]
    fprefix = DATADIR / "source"
    files = {}
    cur_patience = patience
    with tqdm(total=1, desc="Downloading Data") as pbar:
        for i, (collection_id, entity) in enumerate(stream_entities()):
            added_something = False
            try:
                proxy = model.get_proxy(entity)
            except InvalidData:
                continue
            if any(proxy.schema.is_a(s) for s in skip_schemas):
                continue
            for name, category, limit, values in datagen(proxy):
                if name not in files:
                    fname = fprefix / category / f"{name}.txt"
                    pbar.total += sum(f.limit for f in files.values())
                    pbar.refresh()
                    files[name] = SamplesFile(
                        fname=fname, category=category, type=name, limit=limit
                    )
                f = files[name]
                for value in values:
                    if f.add(clean_value(value), collection_id=collection_id):
                        added_something = True
                        pbar.update(1)
            if False and len(files) > 5:
                if not added_something:
                    cur_patience -= 1
                elif added_something:
                    cur_patience = patience
                if all(f.full() for f in files.values()):
                    pbar.write("Done downloading data")
                    break
                elif cur_patience == 0:
                    pbar.write("Out of patience")
                    break
            if i % 10_000 == 0:
                pbar.write("{} {}".format(i, ", ".join(str(f) for f in files.values())))
    filenames = defaultdict(list)
    for f in files.values():
        f.close()
        filenames[f.category].append(f.fname)
    return filenames


def merge_shuffle(filenames, prefix):
    valid = DATADIR / "model_data" / prefix / "valid.txt"
    train = DATADIR / "model_data" / prefix / "train.txt"
    valid.parent.mkdir(parents=True, exist_ok=True)
    train.parent.mkdir(parents=True, exist_ok=True)

    data_valid = []
    data_train = []
    for source in tqdm(filenames, desc="merge shuffle"):
        with open(source) as fd:
            data = list(fd)
            N = int(len(data) * 0.2)
            data_train.extend(data[N:])
            data_valid.extend(data[:N])
    random.shuffle(data_train)
    random.shuffle(data_valid)
    with open(valid, "w+") as fd:
        fd.writelines(data_valid)
    with open(train, "w+") as fd:
        fd.writelines(data_train)
    return train, valid


def evaluate_model(model, data, basedir):
    with open(basedir / "test.json", "w+") as fd:
        result = model.test_label(str(data))
        fd.write(json.dumps(result, indent=4))

    if ANALYSIS_MODULES is False:
        warnings.warn(
            "Analysis modules not installed... skipping confusion matrix. "
            "Install seaborn/pandas/sklearn/matplotlib to create a confusion matrix",
            ImportWarning,
        )
    else:
        y_X = [list(map(str.strip, line.split(" ", 1))) for line in open(data)]
        y, X = zip(*y_X)
        y = list(y)
        X = list(X)
        labels = model.get_labels()
        labels_pretty = [label.replace("__label__", "") for label in labels]
        y_pred = model.predict(X)[0]
        cm = confusion_matrix(y, y_pred, labels=labels, normalize="true")
        df_cm = pd.DataFrame(cm, index=labels_pretty, columns=labels_pretty)
        plt.figure(figsize=(15, 10))
        plt.clf()
        sn.heatmap(df_cm, annot=True, fmt=".2f")
        plt.title("Confusion Matrix")
        plt.ylabel("actual")
        plt.xlabel("predict")
        plt.tight_layout()
        plt.savefig(basedir / "confusion.png")


def fit(train, valid, prefix):
    basedir = DATADIR / "models" / prefix
    basedir.mkdir(parents=True, exist_ok=True)

    ftmodel = fasttext.train_supervised(
        input=str(train), autotuneValidationFile=str(valid), autotuneDuration=600
    )
    evaluate_model(ftmodel, valid, basedir)
    ftmodel.save_model(str(basedir / "model.bin"))

    basedir = basedir / "quantize"
    basedir.mkdir(parents=True, exist_ok=True)

    ftmodel.quantize(verbose=True, input=str(train), retrain=True)
    evaluate_model(ftmodel, valid, basedir)
    ftmodel.save_model(str(basedir / "model.ftz"))


def get_files(t):
    base = DATADIR / "source" / t
    for f in os.listdir(base):
        yield base / f


if __name__ == "__main__":
    print("Downloading data")
    filenames = download_data()
    # filenames = {t: list(get_files(t)) for t in ("types", "schemas")}
    print(filenames)

    print("Creating training data")
    dataset_types = merge_shuffle(filenames["types"], prefix="types")
    dataset_schema = merge_shuffle(filenames["schemas"], prefix="schemas")

    print("Training")
    fit(*dataset_types, prefix="types")
    fit(*dataset_schema, prefix="schemas")
