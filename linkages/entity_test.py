import itertools as IT
import operator
import random
import time
from collections import deque
from zlib import crc32
import warnings
import os

import numpy as np
import ujson
from followthemoney import compare
from followthemoney.exc import InvalidData
from pandas import BooleanDtype, CategoricalDtype, StringDtype


USE_DASK = os.environ.get("FTM_PREDICT_USE_DASK").lower() == "true"
N_LINES_READ = None


if USE_DASK:
    try:
        import dask
        import dask.bag as pipeline
        from dask.cache import Cache
        from dask.distributed import Client, progress
    except ImportError:
        warnings.warn("Dask not found... Using default pipeline", ImportWarning)
        USE_DASK = False
if not USE_DASK:
    import dasklike as pipeline


PHASES = {
    "train": 0.8,
    "test": 0.2,
}

FEATURE_KEYS = [
    "name",
    "country",
    "registrationNumber",
    "incorporationDate",
    "address",
    "jurisdiction",
    "dissolutionDate",
    "mainCountry",
    "ogrnCode",
    "innCode",
    "kppCode",
    "fnsCode",
    "email",
    "phone",
    "website",
    "idNumber",
    "birthDate",
    "nationality",
    "accountNumber",
    "iban",
    "wikidataId",
    "wikipediaUrl",
    "deathDate",
    "cikCode",
    "irsCode",
    "vatCode",
    "okpoCode",
    "passportNumber",
    "taxNumber",
    "bvdId",
]
FEATURE_IDXS = dict(zip(FEATURE_KEYS, range(len(FEATURE_KEYS))))

SCHEMAS = set(
    ("Person", "Company", "LegalEntity", "Organization", "PublicBody", "BankAccount")
)

FIELDS_BAN_LIST = set(["alephUrl", "modifiedAt", "retrievedAt", "sourceUrl"])
DATAFRAME_FIELDS_TYPES = {
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
DATAFRAME_META = {
    f"{which}_{c}": t
    for which in ("left", "right")
    for c, t in DATAFRAME_FIELDS_TYPES.items()
}
DATAFRAME_META["judgement"] = BooleanDtype()
DATAFRAME_META["source"] = CategoricalDtype(["linkage", "negative", "positive"])
DATAFRAME_META["phase"] = CategoricalDtype(PHASES.keys())
DATAFRAME_META["features"] = object


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


def has_properties(entity):
    return "properties" in entity and len(entity["properties"]) >= 1


def has_multi_name(entity):
    return len(set(n for n in entity["properties"].get("name", []) if "@" not in n)) > 1


def entity_to_samples(entity):
    props = {
        k: vs for k, vs in entity.pop("properties").items() if k not in FIELDS_BAN_LIST
    }
    samples = []
    names = set(props.pop("name", []))
    for name in names:
        if props:
            n_props = random.randrange(0, len(props))
            props_keep = random.sample(props.keys(), n_props)
            props_sample = {k: random.choice(props[k]) for k in props_keep}
            samples.append({**entity, "properties": {"name": name, **props_sample}})
        else:
            samples.append({**entity, "properties": {"name": name}})
    return samples


def normalize_linkage(item):
    entity = item.pop("entity")
    return {**item, **entity}


def load_json(item):
    blob, path = item
    data = ujson.loads(blob)
    data["path"] = path
    return data


def create_pairs_positive(path_glob, n_samples=None):
    b = (
        pipeline.read_text(path_glob, include_path=True)
        .map(load_json)
        .filter(has_multi_name)
    )
    if n_samples:
        b = b.take(n_samples, compute=False)
    b = (
        b.map(entity_to_samples)
        .map(pairs_from_group, judgement=True, source="positive", replacement=False)
        .flatten()
    )
    b = b.map(pairs_calc_ftm_features).map(pairs_to_flat)
    return b


def create_pairs_negative(path_glob, n_samples=None):
    b = pipeline.read_text(path_glob, include_path=True).map(load_json)
    if n_samples:
        b = b.take(n_samples, compute=False)
    b = b.map(entity_to_samples).flatten()
    b = (
        b.groupby(operator.itemgetter("path"))
        .map(operator.itemgetter(1))
        .map(pairs_from_group, judgement=False, source="negative", replacement=False)
        .flatten()
    )
    b = b.map(pairs_calc_ftm_features).map(pairs_to_flat)
    return b


def create_pairs_linkages(path_glob, n_samples=None):
    b = (
        pipeline.read_text(path_glob, include_path=True)
        .map(load_json)
        .map(normalize_linkage)
        .filter(has_properties)
    )
    if n_samples:
        b = b.take(n_samples, compute=False)
    b = b.map(entity_to_samples).flatten()
    b = (
        b.groupby(operator.itemgetter("profile_id"), sort=True)
        .map(operator.itemgetter(1))
        .map(
            pairs_from_group,
            judgement=lambda a, b: (
                a.get("decision", False) and b.get("decision", False)
            ),
            replacement=True,
            source="linkage",
        )
        .flatten()
    )
    b = b.map(pairs_calc_ftm_features).map(pairs_to_flat)
    return b


def pairs_to_dataframe(bag):
    return bag.to_dataframe(meta=DATAFRAME_META)


def pairs_from_group(group, judgement, source, replacement=False, max_pairs=5_000_000):
    from followthemoney import model

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
        left, right = group[i], group[j]
        curjudgement = judgement(left, right)
        if curjudgement is False and left["id"] == right["id"]:
            continue
        try:
            schema = model.common_schema(left["schema"], right["schema"])
        except InvalidData:
            continue
        result.append(
            {
                "left": left,
                "right": right,
                "judgement": curjudgement,
                "source": source,
                "schema": schema.name,
                "phase": keys_to_phase(left["id"], right["id"]),
            }
        )
    return result


def pairs_to_flat(item):
    left = item.pop("left")
    right = item.pop("right")
    pair = {
        f"{which}_{k}": v
        for which, entity in (("left", left), ("right", right))
        for k, v in IT.chain(entity["properties"].items(), entity.items())
    }
    return {**item, **pair}


def create_model_proxy(entity, cache={}):
    from followthemoney import model

    eid = entity["id"]
    try:
        return cache[id]
    except KeyError:
        properties = {k: [v] for k, v in entity["properties"].items()}
        cache[eid] = model.get_proxy({**entity, "properties": properties})
        return cache[eid]


def pairs_calc_ftm_features(pair):
    from followthemoney import model

    A = create_model_proxy(pair["left"])
    B = create_model_proxy(pair["right"])
    features = np.zeros(len(FEATURE_KEYS))
    features[FEATURE_IDXS["name"]] = compare.compare_names(A, B)
    features[FEATURE_IDXS["country"]] = compare.compare_countries(A, B)
    schema = model.schemata[pair["schema"]]
    for name, prop in schema.properties.items():
        if name in FIELDS_BAN_LIST:
            continue
        elif prop.name not in FEATURE_IDXS:
            continue
        weight = compare.MATCH_WEIGHTS.get(prop.type)
        if not weight or not prop.matchable:
            continue
        try:
            A_values = A.get(name)
            B_values = B.get(name)
        except InvalidData:
            continue
        if not A_values or not B_values:
            continue
        prop_score = prop.type.compare_sets(A_values, B_values)
        feature_idx = FEATURE_IDXS[prop.name]
        features[feature_idx] = prop_score
    pair["features"] = features
    return pair


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


def onehot_partition(partition, n_classes=2):
    partition = list(partition)
    data = []
    for i, l in enumerate(partition):
        d = [0] * n_classes
        d[l] = 1
        data.append(d)
    return data


if __name__ == "__main__":
    if USE_DASK:
        dask.config.set({"temporary_directory": "/tmp/dask/"})
        cache = Cache(2e9)
        cache.register()
        client = Client(n_workers=1, threads_per_worker=32)
        print(client)

    pairs_linkages = create_pairs_linkages(
        "./data/linkages/linkages-20200803145654.json", n_samples=N_LINES_READ,
    )
    pairs_negative = create_pairs_negative(
        [
            f"./data/entities/raw-latest/legal_entity-{collection.replace(' ', '_')}.json"
            for collection in DEDUPED_COLLECTION_FIDS
        ],
        n_samples=N_LINES_READ,
    )
    pairs_positive = create_pairs_positive(
        "./data/entities/legal_entities-multi_name/*.json", n_samples=N_LINES_READ,
    )

    pairs = pipeline.concat([pairs_linkages, pairs_negative, pairs_positive])
    df = pairs.to_dataframe(meta=DATAFRAME_META)

    parquet_params = {}
    if USE_DASK:
        parquet_params = {"schema": "infer"}
    df.to_parquet("./data/pairs_all.parquet", **parquet_params)
