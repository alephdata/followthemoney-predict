from pathlib import Path
import random
from functools import wraps
from multiprocessing import Process, Queue

import pyarrow as pa
import pyarrow.parquet as pq

import sklearn.utils
import numpy as np
import pandas as pd
import mmh3

from xref_experiments.vocabulary import Vocabulary, stream_vocabulary_record
from xref_experiments.data import entity_generator


RECORD_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("schema", pa.uint8()),
        ("group", pa.uint8()),
        ("property", pa.uint16()),
        ("ngrams", pa.list_(pa.uint32())),
        ("text", pa.string()),
    ]
)


def normalize(phases):
    V = sum(phases.values())
    return {k: v / V for k, v in phases.items()}


def load_vocabularies(basedir):
    basedir = Path(basedir)
    vocabs = {}
    for name in ["ngrams", "group", "property", "schema"]:
        with open(basedir / f"{name}.json", "rb") as fd:
            vocabs[name] = Vocabulary.from_file(fd)
    return vocabs


def resample_dataframe_batch_sizes(fxn):
    @wraps(fxn)
    def _(*args, batch_size=None, batch_last_partial=False, max_samples=None, **kwargs):
        if not batch_size:
            return fxn(*args, **kwargs)
        left_over = None
        n_samples = 0
        for data in fxn(*args, **kwargs):
            if left_over is not None:
                data = pd.concat((left_over, data))
                left_over = None
            while data.shape[0] > batch_size:
                yield data.head(batch_size)
                n_samples += batch_size
                if max_samples and n_samples >= max_samples:
                    return
                data = data.tail(-batch_size)
            left_over = data
        if batch_last_partial:
            yield left_over

    return _


def _multi_skipgram(pmd, queue, *args, **kwargs):
    with pmd:
        for batch in pmd.skipgrams(*args, **kwargs):
            queue.put(batch)
        queue.put(None)


def multiprocess_skipgram(pmd, *args, n_workers=1, queue_max_size=256, **kwargs):
    queue = Queue(queue_max_size)
    process = Process(target=_multi_skipgram, args=(pmd, queue, *args), kwargs=kwargs)
    process.start()
    while True:
        result = queue.get()
        if result is None:
            break
        yield result
    process.join()


class ParquetModelData:
    def __init__(
        self,
        data_dir,
        pa_schema=RECORD_SCHEMA,
        mode="r",
        phases=None,
        vocabularies=None,
        metadata=None,
    ):
        if mode.startswith("w") and vocabularies is None:
            raise ValueError("Must specify vocabulary when writing")
        self.mode = mode
        self.data_dir = Path(data_dir)
        self.metadata = metadata or {}
        self.vocabularies = vocabularies
        self.phases = normalize(phases or {"train": 0.8, "test": 0.10, "valid": 0.10})
        self.pa_schema = pa_schema

    def __enter__(self):
        return self.open()

    def open(self):
        if self.mode.startswith("w"):
            self.__writers_queue = {p: [] for p in self.phases}
            self.__writers = {
                p: pq.ParquetWriter(
                    self.data_dir / f"{p}.parquet", self.pa_schema, compression="SNAPPY"
                )
                for p in self.phases
            }
        elif self.mode.startswith("r"):
            self.__readers = {
                # NOTE: use memmap here?
                p: pq.ParquetFile(self.data_dir / f"{p}.parquet")
                for p in self.phases
            }
        return self

    def __exit__(self, *args, **kwargs):
        return self.close()

    def close(self):
        if self.mode.startswith("w"):
            self.flush_writers(force=True)
            for writer in self.__writers.values():
                writer.close()
        elif self.mode.startswith("r"):
            pass

    def read_groups_shuffle(self, phase, load_n_groups=10, random_seed=None):
        reader = self.__readers[phase]
        n_groups = reader.metadata.num_row_groups
        groups = list(range(n_groups))
        if random_seed:
            random.seed(random_seed)
        random.shuffle(groups)
        for i in range(0, n_groups, load_n_groups):
            row_groups = sorted(groups[i : i + load_n_groups])
            df = pd.concat(
                (reader.read_row_group(g).to_pandas() for g in row_groups),
                copy=False,
            )
            df = df.sort_values("id").reset_index(drop=True)
            yield df

    @resample_dataframe_batch_sizes
    def skipgrams(
        self, phase, negative_sampling=1.0, load_n_groups=10, random_seed=None
    ):
        if random_seed:
            np.random.seed(random_seed)
        for df in self.read_groups_shuffle(phase, load_n_groups):
            # we do a self inner join then remove duplicates. i'd prefer to
            # immediately do comibnations without replacement but i can't find a
            # faster way to do this.
            df["rnd"] = np.random.randint(0, 1 << 16, size=df.shape[0], dtype="uint16")
            pairs_pos = df.merge(
                df.sort_values("rnd"), on="id", how="left", sort=False, copy=False
            ).query("rnd_x != rnd_y")

            pairs_pos["id_y"] = pairs_pos.id
            pairs_pos = pairs_pos.rename(columns={"id": "id_x"}, copy=False).drop(
                columns=["rnd_y", "rnd_x"]
            )

            n_neg = int(pairs_pos.shape[0] * negative_sampling)
            left_neg = sklearn.utils.resample(
                df, n_samples=n_neg, replace=True
            ).reset_index(drop=True)
            right_neg = sklearn.utils.resample(
                df, n_samples=n_neg, replace=True
            ).reset_index(drop=True)
            pairs_neg = left_neg.merge(
                right_neg, left_index=True, right_index=True, copy=False, sort=False
            ).query("id_x != id_y")

            pairs_pos["target"] = 1
            pairs_neg["target"] = 0
            pairs = pd.concat([pairs_pos, pairs_neg], copy=False)
            yield sklearn.utils.shuffle(pairs, random_state=random_seed)

    def flush_writers(self, force=False):
        for phase, queue in self.__writers_queue.items():
            if force or len(queue) >= 32_768:
                data = {field: [] for field in self.pa_schema.names}
                for record in queue:
                    for field, value in record.items():
                        data[field].append(value)
                self.__writers[phase].write_table(pa.table(data, schema=self.pa_schema))
                queue.clear()

    def id_to_phase(self, id_):
        h = mmh3.hash(id_, signed=False) / (1 << 32)
        for phase, v in self.phases.items():
            h -= v
            if h < 0:
                return phase

    def append(self, item):
        ngram = self.vocabularies["ngrams"].metadata["ngram"]
        for proxy, group, prop, tokens in stream_vocabulary_record(
            item, tokenize_args={"ngram": ngram}
        ):
            phase = self.id_to_phase(proxy.id)
            self.__writers_queue[phase].append(
                {
                    "id": proxy.id,
                    "schema": self.vocabularies["schema"][str(proxy.schema.name)],
                    "group": self.vocabularies["group"][str(group)],
                    "property": self.vocabularies["property"][str(prop)],
                    "ngrams": [self.vocabularies["ngrams"][t] for t in tokens],
                    "text": str(proxy.get(prop)),
                }
            )
        self.flush_writers()

    def extend(self, items):
        for item in items:
            self.append(item)


if __name__ == "__main__":
    vocabularies = load_vocabularies(Path("/data/xref-experiments/vocabulary/"))

    data_path = Path("/data/xref-experiments/occrp-data-exports/data/")
    data_exports = list(data_path.glob("*/*.json"))
    items = entity_generator(
        data_exports, entities_per_file=500_000, resevour_sample=True
    )
    # items = stdin_reader()

    basedir = Path("/data/xref-experiments/model-data/")
    basedir.mkdir(exist_ok=True, parents=True)
    with ParquetModelData(basedir, vocabularies=vocabularies, mode="wb+") as pmd:
        for item in items:
            pmd.append(item)
