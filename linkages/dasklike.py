import itertools as IT
from functools import partial, wraps
from pathlib import Path
import os

import pandas as pd
from tqdm import tqdm


DEFAULT_CHUNK_SIZE = 65_536


def dasklike_iter(fxn):
    @wraps(fxn)
    def _(*args, **kwargs):
        return DaskLike(fxn(*args, **kwargs))

    return _


def chunk_stream(stream, chunk_size=None):
    while True:
        data = list(IT.islice(stream, chunk_size))
        if not data:
            return
        yield data


@dasklike_iter
def concat(streams):
    return IT.chain.from_iterable(streams)


@dasklike_iter
def read_text(path_glob, include_path=False, progress=True):
    if not isinstance(path_glob, (list, tuple)):
        path_glob = [path_glob]
    filenames = list(IT.chain.from_iterable(Path().glob(pg) for pg in path_glob))
    if progress:
        path_glob_str = os.path.commonprefix(filenames)
        filenames = tqdm(filenames, desc=f"Reading glob {path_glob_str}", leave=False)
    for filename in filenames:
        with open(filename, "r") as fd:
            if progress:
                fd = tqdm(fd, desc=filename.name, leave=False)
            for line in fd:
                if include_path:
                    yield (line, filename)
                else:
                    yield line


class DaskLike:
    def __init__(self, stream=None):
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
        return DaskLike(data)

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
            df = df.append(df_chunk, ignore_index=True)
        return df
