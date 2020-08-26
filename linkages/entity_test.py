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
from pandas import StringDtype, CategoricalDtype, Int32Dtype, BooleanDtype

dask.config.set({"temporary_directory": "/tmp/dask/"})


PHASES = {
    "train": 0.8,
    "valid": 0.1,
    "test": 0.1,
}

SCHEMAS = set(
    ("Person", "Company", "LegalEntity", "Organization", "PublicBody", "BankAccount")
)

FIELDS_TYPES = {
    "name": StringDtype(),
    "schema": CategoricalDtype(SCHEMAS),
    "id": StringDtype(),
    "country": StringDtype(),
    "address": StringDtype(),
    "registrationNumber": StringDtype(),
    "alias": StringDtype(),
    "status": StringDtype(),
    "classification": StringDtype(),
    "gender": StringDtype(),
    "firstName": StringDtype(),
    "lastName": StringDtype(),
    "birthPlace": StringDtype(),
    "birthDate": StringDtype(),
    "idNumber": StringDtype(),
    "motherName": StringDtype(),
    "nationality": StringDtype(),
}
FIELDS = set(FIELDS_TYPES.keys())
META = {
    f"{which}_{c}": t for which in ("left", "right") for c, t in FIELDS_TYPES.items()
}
META["judgement"] = BooleanDtype()
META["weight"] = "float"
META["phase"] = CategoricalDtype(PHASES.keys())


DEDUPED_COLLECTION_FIDS = set(
    (
        "88513394757c43089cd44f817f16ca05",  # Khadija Project Research Data
        "45602a9bb6c04a179a2657e56ed3a310",  # Mozambique Persons of Interest (2015)
        "zz_occrp_pdi",  # Persona de Interes (2014)
        "ch_seco_sanctions",  # Swiss SECO Sanctions
        "interpol_red_notices",  # INTERPOL Red Notices
        "45602a9bb6c04a179a2657e56ed3a310",
        # "ru_moscow_registration_2014",  # 3.9GB
        "ru_pskov_people_2007",
        "ua_antac_peps",
        "am_voters",
        "hr_gong_peps",
        "hr_gong_companies",
        "mk_dksk",
        "ru_oligarchs",
        "everypolitician",
        "lg_people_companies",
        "rs_prijave",
        "5b5ec30364bb41999f503a050eb17b78",
        "aecf6ecc4ab34955a1b8f7f542b6df62",
        "am_hetq_peps",
        "kg_akipress_peps",
        # "ph_voters",  # 7.5GB
        "gb_coh_disqualified",
    )
)


def pair_combinations(sequence):
    sequence2 = IT.cycle(sequence)
    next(sequence2)
    for i in range(len(sequence) - 1):
        yield from zip(sequence[: -(i + 1)], sequence2)
        deque(IT.islice(sequence2, i + 2), maxlen=0)


def multi_name(entity):
    return len(set(n for n in entity["properties"].get("name", []) if "@" not in n)) > 1


def entity_to_samples(entity, fields=FIELDS):
    props = {k: vs for k, vs in entity.pop("properties").items() if k in fields}
    samples = []
    names = set(props.pop("name", []))
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
    return {**item, **entity}


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
    b = b.map(entity_to_samples).flatten()
    b = (
        b.groupby(operator.itemgetter("id"))
        .map(operator.itemgetter(1))
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
        .map(pairs_from_group, judgement=False, weight=weight, replacement=False)
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


def pairs_from_group(
    group, judgement, weight=0.5, replacement=False, max_pairs=5_000_000
):
    N = len(group)
    if N < 2:
        return []
    if judgement in (True, False):
        __judgement = judgement
        judgement = lambda a, b: __judgement
    idxs = list(range(N))
    random.shuffle(idxs)
    if replacement:
        indicies = pair_combinations(idxs)
    else:
        indicies = zip(idxs[N // 2 :], idxs[: N // 2])
    result = []
    for i, j in indicies:
        if len(result) >= max_pairs:
            break
        curjudgement = judgement(group[i], group[j])
        if curjudgement is False and group[i]["id"] == group[j]["id"]:
            continue
        pair = {
            f"{which}_{k}": v
            for which, sample in (("left", group[i]), ("right", group[j]))
            for k, v in sample.items()
        }
        pair["judgement"] = curjudgement
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


def compute(client, task, desc=None):
    if desc:
        print(desc)
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
    # client = Client(n_workers=4, threads_per_worker=1, memory_limit="2GB")
    client = Client(n_workers=1, threads_per_worker=12)
    print(client)

    pairs_linkages = create_pairs_linkages(
        "./data/linkages/linkages-20200803145654.json"
    )
    pairs_negative = create_pairs_negative(
        [
            f"./data/entities/raw-latest/legal_entity-{collection.replace(' ', '_')}.json"
            for collection in DEDUPED_COLLECTION_FIDS
        ]
    )
    pairs_positive = create_pairs_positive(
        "./data/entities/legal_entities-multi_name/*.json"
    )

    # n_pairs_linkages = compute(
    #    client, pairs_linkages.count(), desc="Calculating Linkages"
    # )
    # n_pairs_negative = compute(
    #    client, pairs_negative.count(), desc="Calculating Negatives"
    # )
    # n_pairs_positive = compute(
    #    client, pairs_positive.count(), desc="Calculating Positives"
    # )
    # N = n_pairs_linkages + n_pairs_positive + n_pairs_negative
    # print(
    #    f"linkages: {n_pairs_linkages}, positive: {n_pairs_positive}, negative: {n_pairs_negative}"
    # )

    pairs = dask.bag.concat([pairs_linkages, pairs_negative, pairs_positive])
    df = pairs_to_dataframe(pairs)
    df.to_parquet("./data/entity_test.parquet", schema="infer")
    # df = compute(client, ddf.persist(), desc="Pinning DataFrame to memory")

    # n_positive = compute(client, df.judgement.sum(), desc="Counting Positives")
    # print(f"n_pairs: {N}, n_positive: {n_positive}, pct: {n_positive / N}")
    # print(df.head(10))

    # df = bag.to_dataframe()
    # print(df.head(10))
