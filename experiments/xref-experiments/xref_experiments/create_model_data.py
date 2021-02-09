from pathlib import Path
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
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
    ]
)


def normalize(phases):
    V = sum(phases.values())
    return {k: v / V for k, v in phases.items()}


def load_vocabularies(basedir):
    vocabs = {}
    for name in ["ngrams", "group", "property", "schema"]:
        with open(basedir / f"{name}.json", "rb") as fd:
            vocabs[name] = Vocabulary.from_file(fd)
    return vocabs


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
        if self.mode.startswith("w"):
            self.__writers_queue = {p: [] for p in self.phases}
            self.__writers = {
                p: pq.ParquetWriter(
                    self.data_dir / f"{p}.parquet", self.pa_schema, compression="SNAPPY"
                )
                for p in self.phases
            }
        return self

    def __exit__(self, *args, **kwargs):
        if self.mode.startswith("w"):
            self.flush_writers(force=True)
            for writer in self.__writers.values():
                writer.close()

    def flush_writers(self, force=False):
        for phase, queue in self.__writers_queue.items():
            if force or len(queue) >= 65_536:
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
    items = entity_generator(data_exports, entities_per_file=1_000_000)
    # items = stdin_reader()

    basedir = Path("/data/xref-experiments/model-data/")
    basedir.mkdir(exist_ok=True, parents=True)
    with ParquetModelData(basedir, vocabularies=vocabularies, mode="wb+") as pmd:
        for item in items:
            pmd.append(item)
