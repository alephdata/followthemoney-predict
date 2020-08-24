import dask
import dask.bag
from dask.distributed import Client, progress
import ujson
import random
from zlib import crc32
import operator

from pandas.api.types import CategoricalDtype
from normality import normalize


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
        .map(pairs_from_group, judgement=True, weight=weight)
        .flatten()
    )
    return b


def create_pairs_negative(path_glob, weight=0.5):
    b = dask.bag.read_text(path_glob, include_path=True).map(load_json)
    b = (
        b.map(entity_to_samples)
        .flatten()
        .groupby(operator.itemgetter("path"))
        .map(operator.itemgetter(1))
        .map(pairs_from_group, judgement=False, weight=weight)
        .flatten()
    )
    return b


def pairs_to_dataframe(bag, field_pct=0.05):
    return bag.to_dataframe(meta=META)


def pairs_from_group(group, judgement, weight=0.5):
    N = len(group)
    if N < 2:
        return []
    idxs = list(range(len(group)))
    random.shuffle(idxs)
    result = []
    for i, j in zip(idxs[N // 2 :], idxs[: N // 2]):
        pair = {
            f"{which}_{k}": v
            for which, sample in (("left", group[i]), ("right", group[j]))
            for k, v in sample.items()
        }
        pair["judgement"] = True
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
    progress(f)
    return client.gather(f)


if __name__ == "__main__":
    client = Client(n_workers=8, threads_per_worker=1, memory_limit="1GB")
    print(client)

    pairs_positive = create_pairs_positive(
        "./data/entities/legal_entities-multi_name/*.json"
    )
    pairs_negative = create_pairs_negative("./data/collections/*.json")

    n_positive = compute(client, pairs_positive.count())
    n_negative = compute(client, pairs_negative.count())
    print(f"positive: {n_positive}, negative: {n_negative}")

    pairs = dask.bag.concat([pairs_negative, pairs_positive])
    df = pairs_to_dataframe(pairs)
    print(df.head(10))

    # df = bag.to_dataframe()
    # print(df.head(10))
