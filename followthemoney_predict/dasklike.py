import gzip
import itertools as IT
import logging
import os
import random
from functools import partial, wraps
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

DEFAULT_CHUNK_SIZE = 8_192


def init():
    print("Using DaskLike")


def from_sequence(iter_, *args, **kwargs):
    return DaskLike(iter_)


def dasklike_iter(fxn):
    @wraps(fxn)
    def _(*args, **kwargs):
        return from_sequence(fxn(*args, **kwargs))

    return _


def chunk_stream(stream, chunk_size=DEFAULT_CHUNK_SIZE):
    while True:
        data = list(IT.islice(stream, chunk_size))
        if not data:
            return
        yield data


@dasklike_iter
def concat(streams):
    return IT.chain.from_iterable(streams)


@dasklike_iter
def read_text(path_glob, include_path=False, compression=None, progress=True):
    if not isinstance(path_glob, (list, tuple)):
        path_glob = [path_glob]
    filenames = list(IT.chain.from_iterable(Path().glob(str(pg)) for pg in path_glob))
    if progress:
        path_glob_str = os.path.commonprefix(filenames)
        filenames = tqdm(filenames, desc=f"Reading glob {path_glob_str}", leave=False)
    for filename in filenames:
        if filename.suffix.endswith(".gz") or compression == "gzip":
            openfxn = gzip.open
        else:
            openfxn = open
        with openfxn(filename, "r") as fd:
            if progress:
                fd = tqdm(fd, desc=filename.name, leave=False)
            for line in fd:
                if include_path:
                    yield (line, filename)
                else:
                    yield line


class DaskLike:
    def __init__(self, stream=None):
        if isinstance(stream, DaskLike):
            self.stream = stream.stream
        else:
            self.stream = stream

    def __iter__(self):
        return iter(self.stream)

    @dasklike_iter
    def map(self, fxn, *args, **kwargs):
        return map(partial(fxn, *args, **kwargs), self.stream)

    @dasklike_iter
    def flatten(self):
        return IT.chain.from_iterable(self.stream)

    @dasklike_iter
    def map_partitions(self, fxn, *args, partition_size=DEFAULT_CHUNK_SIZE, **kwargs):
        fxn_partial = partial(fxn, *args, **kwargs)
        for chunk in chunk_stream(self.stream, chunk_size=partition_size):
            yield map(fxn_partial, self.stream)

    @dasklike_iter
    def groupby(self, grouper, sort=False):
        stream = self.stream
        if sort:
            stream = list(stream)
            stream.sort(key=grouper)
        return ((k, list(g)) for k, g in IT.groupby(stream, key=grouper))

    @dasklike_iter
    def filter(self, fxn):
        return filter(fxn, self.stream)

    def take(self, N, compute=True):
        take_stream, self.stream = IT.tee(self.stream)
        data = IT.islice(take_stream, N)
        if compute:
            return list(data)
        return from_sequence(data)

    @dasklike_iter
    def debug_counter(self, desc, every=500):
        for i, item in enumerate(self.stream):
            if (i + 1) % every == 0:
                print(f"[{desc}] {i}")
            yield item

    @dasklike_iter
    def debug_sampler(self, desc, proba):
        for item in self.stream:
            if random.random() < proba:
                print(f"[{desc}] Sample:")
                print(item)
            yield item

    def to_dataframe(
        self, meta=None, columns=None, partition_size=DEFAULT_CHUNK_SIZE, progress=True
    ):
        if meta:
            columns = list(meta.keys())
        stream = self.stream
        if progress:
            stream = tqdm(stream, desc="Creating Dataframe")
        df = pd.DataFrame(columns=columns)
        if meta:
            df = df.astype(meta)
        for chunk in chunk_stream(stream, chunk_size=partition_size):
            df_chunk = pd.DataFrame(chunk, columns=columns)
            if meta:
                df_chunk = df_chunk.astype(meta)
            df = df.append(df_chunk, ignore_index=True)
        return df

    def to_parquet(
        self, filename, meta, partition_size=DEFAULT_CHUNK_SIZE, progress=True
    ):
        writer = None
        columns = list(meta.keys())
        chunks = chunk_stream(self.stream, chunk_size=partition_size)
        if progress:
            chunks = tqdm(
                chunks, desc="Streaming to Parquet", unit_scale=partition_size
            )
        for chunk in chunks:
            df_chunk = pd.DataFrame(chunk, columns=columns).astype(meta)
            if writer is None:
                schema = pa.Schema.from_pandas(df_chunk, preserve_index=False)
                writer = pq.ParquetWriter(filename, schema)
            table = pa.Table.from_pandas(df_chunk, schema=schema)
            writer.write_table(table)
        writer.close()
        return writer
