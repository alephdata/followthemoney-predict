import itertools as IT
import operator
import random
from pathlib import Path
from zlib import crc32

import numpy as np
from followthemoney import compare
from followthemoney.exc import InvalidData

from .pipeline import pipeline
from . import const
from .util import pair_combinations

N_LINES_READ = 100
DEBUG = False


def has_properties(entity):
    return "properties" in entity and len(entity["properties"]) >= 1


def has_multi_name(entity):
    return len(set(n for n in entity["properties"].get("name", []) if "@" not in n)) > 1


def entity_to_samples(entity, field_ban=const.FIELDS_BAN_SET):
    props = {
        k: vs for k, vs in entity.pop("properties").items() if vs and k not in field_ban
    }
    entity["collection_id"] = str(entity["collection_id"])
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


def normalize_profile(item):
    entity = item.pop("entity")
    return {
        "judgement": item["judgement"],
        "entityset_id": item["entityset_id"],
        "collection_id": item["collection"]["collection_id"],
        **entity,
    }


def create_pairs_positive(data_stream, n_lines_read=None):
    b = data_stream.filter(has_properties)
    if n_lines_read:
        b = b.take(n_lines_read, compute=False)
    if DEBUG:
        b = b.debug_counter("Positive Stream").debug_sampler("Positive Stream", 0.001)
    b = (
        b.filter(has_multi_name)
        .map(entity_to_samples)
        .map(pairs_from_group, judgement=True, source="positive", replacement=False)
        .flatten()
    )
    if DEBUG:
        b = b.debug_counter("Positive Pairs").debug_sampler("Positive Pairs", 0.01)
    b = b.map(pairs_calc_ftm_features).map(pairs_to_flat)
    return b


def create_pairs_negative(data_stream, n_lines_read=None):
    b = data_stream.filter(has_properties)
    if n_lines_read:
        b = b.take(n_lines_read, compute=False)
    if DEBUG:
        b = b.debug_counter("Negative Stream").debug_sampler("Negative Stream", 0.001)
    b = b.map(entity_to_samples).flatten()
    b = (
        b.groupby(operator.itemgetter("collection_id"))
        .map(operator.itemgetter(1))
        .map(pairs_from_group, judgement=False, source="negative", replacement=False)
        .flatten()
    )
    if DEBUG:
        b = b.debug_counter("Negative Pairs").debug_sampler("Negative Pairs", 0.01)
    b = b.map(pairs_calc_ftm_features).map(pairs_to_flat)
    return b


def create_pairs_profile(data_stream, n_lines_read=None):
    judgement_map = {
        "negative": False,
        "positive": True,
    }
    b = data_stream.map(normalize_profile)
    if n_lines_read:
        b = b.take(n_lines_read, compute=False)
    if DEBUG:
        b = b.debug_counter("Profile Stream").debug_sampler("Profile Stream", 0.001)
    b = b.filter(has_properties).map(entity_to_samples).flatten()
    b = (
        b.groupby(operator.itemgetter("entityset_id"))
        .map(operator.itemgetter(1))
        .map(
            pairs_from_group,
            judgement=lambda a, b: (
                judgement_map.get(a["judgement"], False)
                and judgement_map.get(b["judgement"], False)
            ),
            replacement=True,
            source="profile",
        )
        .flatten()
    )
    if DEBUG:
        b = b.debug_counter("Profile Pairs").debug_sampler("Profile Pairs", 0.01)
    b = b.map(pairs_calc_ftm_features).map(pairs_to_flat)
    return b


def pairs_to_dataframe(bag):
    return bag.to_dataframe(meta=const.DATAFRAME_META)


def pairs_from_group(group, judgement, source, replacement=False, max_pairs=5_000_000):
    from followthemoney import model

    N = len(group)
    if N < 2:
        return []
    if judgement in (True, False):
        __judgement = judgement
        judgement = lambda a, b: __judgement  # NOQA
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


def pairs_calc_ftm_features(
    pair, feature_idxs=const.FEATURE_IDXS, fields_ban=const.FIELDS_BAN_SET
):
    from followthemoney import model

    A = create_model_proxy(pair["left"])
    B = create_model_proxy(pair["right"])
    features = np.zeros(len(feature_idxs))
    features[feature_idxs["name"]] = compare.compare_names(A, B)
    features[feature_idxs["country"]] = compare.compare_countries(A, B)
    schema = model.schemata[pair["schema"]]
    for name, prop in schema.properties.items():
        if name in fields_ban:
            continue
        elif prop.name not in feature_idxs:
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
        feature_idx = feature_idxs[prop.name]
        features[feature_idx] = prop_score
    pair["features"] = features
    return pair


def keys_to_phase(key_a, key_b):
    if key_b > key_a:
        return keys_to_phase(key_b, key_a)
    i = float(crc32(f"{key_a}:{key_b}".encode("utf8")) & 0xFFFFFFFF) / 2 ** 32
    for phase, c in const.PHASES.items():
        i -= c
        if i <= 0:
            return phase


def create_full_stream(
    stream_set, meta=const.DATAFRAME_META, n_lines_read=N_LINES_READ
):
    pairs_profile = create_pairs_profile(stream_set.profile, n_lines_read=n_lines_read)
    pairs_negative = create_pairs_negative(
        stream_set.negative, n_lines_read=n_lines_read
    )
    pairs_positive = create_pairs_positive(
        stream_set.positive, n_lines_read=n_lines_read
    )

    pairs = pipeline.concat([pairs_profile, pairs_negative, pairs_positive])
    return pairs


if __name__ == "__main__":
    from .data_sources import DATA_SOURCES

    METHOD = "aleph"
    parquet_path = (
        Path(__file__).parent / f"data/pairs.{METHOD}.{N_LINES_READ or 'all'}.parquet"
    ).relative_to(Path.cwd())

    print("Using method:", METHOD)
    print("n lines to read:", N_LINES_READ)
    print("Writing to:", parquet_path)

    pipeline.init()
    stream_set = DATA_SOURCES[METHOD].get_data_streams(pipeline)
    pairs = create_full_stream(stream_set, n_lines_read=N_LINES_READ)

    if pipeline.IS_DASK:
        pairs = create_full_stream(stream_set)
        df = pairs.to_dataframe(meta=const.DATAFRAME_META)
        df.to_parquet(parquet_path, schema="infer")
    else:
        pairs.to_parquet(parquet_path, meta=const.DATAFRAME_META)
