from pathlib import Path
import random

import orjson as json
from tqdm.autonotebook import tqdm


def load_file_stream(
    path, collection_foreign_id=None, n_entities=None, line_limit=4096
):
    with tqdm(
        total=path.stat().st_size,
        desc=collection_foreign_id,
        leave=False,
        unit_divisor=1024,
        unit="B",
        unit_scale=True,
    ) as pbar:
        N = 0
        with open(path, "rb") as fd:
            for line in fd:
                pbar.update(fd.tell() - pbar.n)
                if len(line) > line_limit:
                    continue
                data = json.loads(line)
                if collection_foreign_id:
                    data["collection_foreign_id"] = collection_foreign_id
                N += 1
                yield data
                if n_entities and N >= n_entities:
                    return


def load_file_resevour(
    path, collection_foreign_id=None, n_entities=500_000, line_limit=4096
):
    with tqdm(
        total=path.stat().st_size,
        desc=collection_foreign_id,
        leave=False,
        unit_divisor=1024,
        unit="B",
        unit_scale=True,
    ) as pbar:
        N = 0
        samples = []
        with open(path, "rb") as fd:
            for line in fd:
                pbar.update(fd.tell() - pbar.n)
                if len(line) > line_limit:
                    continue
                if len(samples) >= n_entities:
                    i = random.randint(0, N)
                    if i < len(samples):
                        samples[i] = line
                else:
                    samples.append(line)
                N += 1
        for line in samples:
            data = json.loads(line)
            if collection_foreign_id:
                data["collection_foreign_id"] = collection_foreign_id
            yield data


def entity_generator(data_paths, entities_per_file=None, resevour_sample=False):
    data_paths.sort(key=lambda p: p.stat().st_size)
    total = sum(p.stat().st_size for p in data_paths)
    load_file = load_file_resevour if resevour_sample else load_file_stream
    with tqdm(
        total=total, desc="Exports", unit_divisor=1024, unit="B", unit_scale=True
    ) as pbar:
        for path in data_paths:
            path = Path(path)
            cfid = path.parent.name
            entities = load_file(path, cfid, n_entities=entities_per_file)
            yield from entities
            pbar.update(path.stat().st_size)
