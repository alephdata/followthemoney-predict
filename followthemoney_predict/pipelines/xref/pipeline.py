import itertools as IT
import operator
import random
from collections import deque
from zlib import crc32

import numpy as np
from banal import ensure_list
from followthemoney import compare
from followthemoney.exc import InvalidData

from . import settings

PROXY_CACHE_SIZE = 1_000_000
N_LINES_READ = 100
DEBUG = False


def has_properties(entity):
    return "properties" in entity and len(entity["properties"]) >= 1


def has_multi_name(entity):
    return len(set(n for n in entity["properties"].get("name", []) if "@" not in n)) > 1


def pair_combinations(sequence):
    sequence2 = IT.cycle(sequence)
    next(sequence2)
    for i in range(len(sequence) - 1):
        yield from zip(sequence[: -(i + 1)], sequence2)
        deque(IT.islice(sequence2, i + 2), maxlen=0)


def entity_to_samples(entity, field_ban=settings.FIELDS_BAN_SET):
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


def create_dataframe_from_entities(data_stream, meta=None, source="predict"):
    data_stream = (
        data_stream.map(make_pair, judgement=None, source=source)
        .filter(None)
        .map(pairs_calc_ftm_features)
        .map(pairs_to_flat)
    )
    df = data_stream.to_dataframe(meta=meta)
    return df


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


def pairs_from_group(group, judgement, source, replacement=False, max_pairs=5_000_000):
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
        pair = make_pair((left, right), curjudgement, source)
        if pair is not None:
            result.append(pair)
    return result


def make_pair(pair, judgement, source):
    from followthemoney import model

    (left, right) = pair
    if judgement is False and left["id"] == right["id"]:
        return None
    try:
        schema = model.common_schema(left["schema"], right["schema"])
    except InvalidData:
        return None
    return {
        "left": left,
        "right": right,
        "judgement": judgement,
        "schema": schema.name,
        "source": source,
    }


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
        return cache[eid]
    except KeyError:
        properties = {k: ensure_list(v) for k, v in entity["properties"].items()}
        c = cache[eid] = model.get_proxy({**entity, "properties": properties})
        if len(cache) > PROXY_CACHE_SIZE:
            N = len(cache) - PROXY_CACHE_SIZE
            keys = IT.islice(cache.keys(), N)
            for k in keys:
                cache.pop(k)
        return c


def pairs_calc_ftm_features(
    pair, feature_idxs=settings.FEATURE_IDXS, fields_ban=settings.FIELDS_BAN_SET
):
    A = create_model_proxy(pair["left"])
    B = create_model_proxy(pair["right"])
    pair["features"] = ftm_features_from_proxy(
        A, B, pair["schema"], feature_idxs, fields_ban
    )
    return pair


def ftm_features_from_proxy(
    A, B, schema, feature_idxs=settings.FEATURE_IDXS, fields_ban=settings.FIELDS_BAN_SET
):
    from followthemoney import model

    features = np.zeros(len(feature_idxs))
    features[feature_idxs["name"]] = compare.compare_names(A, B)
    features[feature_idxs["country"]] = compare.compare_countries(A, B)
    schema = model.schemata[schema]
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
    return features


def keys_to_phase(key_a, key_b, phase):
    if key_b > key_a:
        return keys_to_phase(key_b, key_a, phase)
    i = float(crc32(f"{key_a}:{key_b}".encode("utf8")) & 0xFFFFFFFF) / 2 ** 32
    for phase, c in phase.items():
        i -= c
        if i <= 0:
            return phase


def calculate_phase(entity, phases):
    entity["phase"] = keys_to_phase(entity["left_id"], entity["right_id"], phases)
    return entity


def create_full_stream(workflow, stream_set, n_lines_read=N_LINES_READ, phases=None):
    pairs_profile = create_pairs_profile(stream_set.profile, n_lines_read=n_lines_read)
    pairs_negative = create_pairs_negative(
        stream_set.negative, n_lines_read=n_lines_read
    )
    pairs_positive = create_pairs_positive(
        stream_set.positive, n_lines_read=n_lines_read
    )

    pairs = workflow.concat([pairs_profile, pairs_negative, pairs_positive])

    if phases:
        pairs = pairs.map(calculate_phase, phases=phases)
    return pairs
