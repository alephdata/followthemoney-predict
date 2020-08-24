import itertools as IT
from collections import deque
import operator
import random
import time
from zlib import crc32

import dask
import dask.bag
import ujson
from dask.distributed import Client, progress
from normality import normalize
from pandas.api.types import CategoricalDtype

dask.config.set({"temporary_directory": "/tmp/dask/"})


PHASES = {
    "train": 0.8,
    "valid": 0.1,
    "test": 0.1,
}

FIELDS = {
    "name",
    "schema",
    "id",
    "country",
    "address",
    "registrationNumber",
    "alias",
    "status",
    "classification",
    "gender",
    "firstName",
    "lastName",
    "birthPlace",
    "birthDate",
    "idNumber",
    "motherName",
    "nationality",
}
META = {f"{which}_{c}": str for which in ("left", "right") for c in FIELDS}
META["judgement"] = bool
META["weight"] = int
META["phase"] = CategoricalDtype(PHASES.keys())


def pair_combinations(sequence):
    sequence2 = IT.cycle(sequence)
    next(sequence2)
    for i in range(len(sequence) - 1):
        yield from zip(sequence[: -(i + 1)], sequence2)
        deque(IT.islice(sequence2, i + 2), maxlen=0)


def multi_name(entity):
    return len(set(n for n in entity["properties"].get("name", []) if "@" not in n)) > 1


def entity_to_samples(entity, fields=FIELDS):
    props = {
        k: list(set(normalize(v, latinize=True) for v in vs))
        for k, vs in entity.pop("properties").items()
        if k in fields
    }
    samples = []
    names = set(normalize(n, latinize=True) for n in props.pop("name", []))
    for name in names:
        if props:
            n_props = random.randrange(0, len(props))
            props_keep = random.sample(props.keys(), n_props)
            props_sample = {k: random.choice(props[k]) for k in props_keep}
            samples.append({"name": name, **entity, **props_sample})
        else:
            samples.append({"name": name, **entity})
    return samples


def normalize_linkage(item):
    entity = item.pop("entity")
    return {**entity, **item}


def load_json(item):
    blob, path = item
    data = ujson.loads(blob)
    data["path"] = path
    return data


def create_pairs_positive(path_glob, weight=0.5):
    b = (
        dask.bag.read_text(path_glob, include_path=True)
        .map(load_json)
        .filter(multi_name)
    )
    b = (
        b.map(entity_to_samples)
        .map(pairs_from_group, judgement=True, weight=weight, replacement=False)
        .flatten()
    )
    return b


def create_pairs_negative(path_glob, weight=0.5):
    b = dask.bag.read_text(path_glob, include_path=True).map(load_json)
    b = b.map(entity_to_samples).flatten()
    b = (
        b.groupby(operator.itemgetter("path"))
        .map(operator.itemgetter(1))
        .map(pairs_from_group, judgement=False, weight=weight, replacement=True)
        .flatten()
    )
    return b


def create_pairs_linkages(path_glob, weight=1.0):
    b = (
        dask.bag.read_text(path_glob, include_path=True)
        .map(load_json)
        .map(normalize_linkage)
    )
    b = b.map(entity_to_samples).flatten()
    b = (
        b.groupby(operator.itemgetter("profile_id"))
        .map(operator.itemgetter(1))
        .map(
            pairs_from_group,
            judgement=lambda a, b: (
                a.get("decision", False) and b.get("decision", False)
            ),
            replacement=True,
            weight=weight,
        )
        .flatten()
    )
    return b


def pairs_to_dataframe(bag, field_pct=0.05):
    return bag.to_dataframe(meta=META)


def pairs_from_group(group, judgement, weight=0.5, replacement=False):
    N = len(group)
    if N < 2:
        return []
    if judgement in (True, False):
        __judgement = judgement
        judgement = lambda a, b: __judgement
    idxs = list(range(N))
    random.shuffle(idxs)
    if replacement:
        indicies = IT.islice(pair_combinations(idxs), 1_000_000)
    else:
        indicies = zip(idxs[N // 2 :], idxs[: N // 2])
    result = []
    for i, j in indicies:
        pair = {
            f"{which}_{k}": v
            for which, sample in (("left", group[i]), ("right", group[j]))
            for k, v in sample.items()
        }
        pair["judgement"] = judgement(group[i], group[j])
        pair["weight"] = weight
        pair["phase"] = keys_to_phase(pair["left_id"], pair["right_id"])
        result.append(pair)
    return result


def keys_to_phase(key_a, key_b):
    if key_b > key_a:
        return keys_to_phase(key_b, key_a)
    i = float(crc32(f"{key_a}:{key_b}".encode("utf8")) & 0xFFFFFFFF) / 2 ** 32
    for phase, c in PHASES.items():
        i -= c
        if i <= 0:
            return phase


def compute(client, task):
    f = client.compute(task)
    while True:
        try:
            progress(f)
            break
        except OSError:
            print("Timeout starting progressbar... trying again")
            time.sleep(1)
    print()
    return client.gather(f)


if __name__ == "__main__":
    client = Client(n_workers=8, threads_per_worker=1, memory_limit="3GB")
    print(client)

    pairs_linkages = create_pairs_linkages(
        "./data/linkages/linkages-20200803145654.json"
    )
    pairs_positive = create_pairs_positive(
        "./data/entities/legal_entities-multi_name/*.json"
    )
    pairs_negative = create_pairs_negative("./data/collections/*.json")

    n_pairs_linkages = compute(client, pairs_linkages.count())
    n_pairs_positive = compute(client, pairs_positive.count())
    n_pairs_negative = compute(client, pairs_negative.count())
    N = n_pairs_linkages + n_pairs_positive + n_pairs_negative
    print(
        f"linkages: {n_pairs_linkages}, positive: {n_pairs_positive}, negative: {n_pairs_negative}"
    )

    pairs = dask.bag.concat([pairs_linkages, pairs_negative, pairs_positive])
    df = pairs_to_dataframe(pairs).repartition(npartitions=N // 10000)
    n_positive = compute(client, df.judgement.sum())
    print(f"n_pairs: {N}, n_positive: {n_positive}, pct: {n_positive / N}")
    print(df.head(10))

    # df = bag.to_dataframe()
    # print(df.head(10))
