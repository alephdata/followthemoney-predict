"""
This file needs a lot of refactoring. It currently is a concatination of
multiple scripts with the thinnest of wrappers with the Sampler class which at
least provides a decent API for the rest of the module.

Most of this can be replaced with Pandas and a simpler wrapper class.
"""
from collections import defaultdict, Counter
from itertools import combinations, product, islice, groupby
import math
import random

from tqdm import tqdm
import numpy as np
from banal import ensure_list
from followthemoney.exc import InvalidData
from followthemoney import model
from followthemoney import compare

import util


FIELDS_BAN_LIST = set(["alephUrl", "modifiedAt", "retrievedAt", "sourceUrl"])


class Sampler:
    def __init__(self, links):
        features_entities, targets_raw, profiles_raw = zip(*emit_pairs(links))
        X, features, ftm_scores, fields, schemas, indicies = extract_data(
            features_entities
        )

        targets = np.asarray([targets_raw[i] for i in indicies])
        profiles = [profiles_raw[i] for i in indicies]

        (
            features_train,
            features_test,
            X_train,
            X_test,
            y_train,
            y_test,
            ftm_train,
            ftm_test,
            schemas_train,
            schemas_test,
            indicies_train,
            indicies_test,
        ) = train_test_split(
            features,
            X,
            targets,
            ftm_scores,
            schemas,
            list(range(X.shape[0])),
            test_pct=0.25,
            correlation_marker=profiles,
            # balance_classes=targets,
        )

        l = locals()
        self.data = {
            d: l[d]
            for d in (
                "features_train",
                "features_test",
                "X_train",
                "X_test",
                "y_train",
                "y_test",
                "ftm_train",
                "ftm_test",
                "schemas_train",
                "schemas_test",
                "indicies_train",
                "indicies_test",
            )
        }
        self.data["fields"] = fields
        self.data["schemas"] = schemas

    def __getattribute__(self, key):
        try:
            return super().__getattribute__(key)
        except AttributeError:
            return self.data[key]

    def summarize(self):
        print(
            f"Training samples: {len(self.X_train)}. Test samples: {len(self.X_test)}. Features: {self.X_train.shape[1]}"
        )
        print(
            f"Pct Positive Training: {sum(self.y_train) / len(self.y_train) * 100:0.2f}% "
            f"Pct Positive Testing: {sum(self.y_test) / len(self.y_test) * 100:0.2f}%"
        )
        print("=" * 20)


def create_collection_pseudopairs(entities):
    comparisons = {}
    num_properties = Counter()
    entities_lookup = {e["id"]: e for e in entities}
    proxies = [model.get_proxy(e) for e in entities]
    for A in proxies:
        num_properties[A.id] = sum(len(p) for p in A.properties) / len(A.properties)
    N = math.factorial(len(proxies)) / (math.factorial(len(proxies) - 2) * 2)
    with tqdm(total=N, desc="Comparing collection") as pbar:
        for i, A in enumerate(proxies[:-1]):
            best_score = 0.1
            best_key = None
            for B in proxies[i + 1 :]:
                key = [A.id, B.id]
                key.sort()
                score = compare.compare(model, A, B)
                pbar.update(1)
                if score > best_score:
                    best_score = score
                    best_key = tuple(key)
            if best_key:
                comparisons[tuple(key)] = score
                pbar.set_description(
                    f"Comparing collection ({len(comparisons)} candidates)"
                )
    while comparisons:
        (key,) = random.choices(
            list(comparisons.keys()), weights=list(comparisons.values())
        )
        A_id, B_id = key
        comparisons.pop(key)
        profile_id = f"{A_id}_{B_id}"
        yield {
            "profile_id": profile_id,
            "decision": False,
            "entity_id": A_id,
            "entity": entities_lookup[A_id],
        }
        yield {
            "profile_id": profile_id,
            "decision": False,
            "entity_id": B_id,
            "entity": entities_lookup[B_id],
        }
        if num_properties:
            (A_id,) = random.choices(
                list(num_properties.keys()), weights=list(num_properties.values())
            )
            num_properties.pop(A_id)
            A = entities_lookup[A_id]
            A_entities = islice(create_split_entity(A), 5)
            for A_entity in A_entities:
                yield {
                    "profile_id": A_id,
                    "decision": True,
                    "entity_id": A_id,
                    "entity": A_entity,
                }


def create_split_proxies(data):
    for split in create_split_entity(data):
        yield model.get_proxy(split)


def create_split_entity(data):
    properties = {
        k: ensure_list(v)
        for k, v in data["properties"].items()
        if k not in FIELDS_BAN_LIST
    }
    fields = properties.keys()
    for p in properties.values():
        random.shuffle(p)
    for values in product(*properties.values()):
        new_properties = {f: [v] for f, v in zip(fields, values)}
        yield {**data, "properties": new_properties}


def emit_pairs(
    links,
    balance_classes=False,
    split_entities=True,
    num_split_entities=10,
    max_profile_links=5,
):
    profiles = defaultdict(list)
    unused_entities = set()
    for link in links:
        profile = link["profile_id"]
        profiles[profile].append(link)
        unused_entities.add((profile, link["entity_id"]))
    class_counts = Counter()
    for profile, links in tqdm(profiles.items(), desc="Creating pairs"):
        if max_profile_links and len(links) > max_profile_links:
            links = random.sample(links, max_profile_links)
        if len(links) > 2:
            for A, B in combinations(links, 2):
                decision = A["decision"] and B["decision"]
                unused_entities.discard((profile, A["entity_id"]))
                unused_entities.discard((profile, B["entity_id"]))
                if split_entities:
                    A_entities = create_split_proxies(A["entity"])
                    B_entities = create_split_proxies(B["entity"])
                    for A_entity, B_entity in product(
                        islice(A_entities, num_split_entities),
                        islice(B_entities, num_split_entities),
                    ):
                        class_counts[decision] += 1
                        yield (A_entity, B_entity), decision, profile
                else:
                    class_counts[decision] += 1
                    yield (
                        (model.get_proxy(A["entity"]), model.get_proxy(B["entity"])),
                        decision,
                        profile,
                    )
    if balance_classes:
        with tqdm(
            desc="Balancing Classes", total=class_counts[False] - class_counts[True],
        ) as pbar:
            while class_counts[True] < class_counts[False]:
                if unused_entities:
                    profile, entity_id = unused_entities.pop()
                    link = next(
                        l for l in profiles[profile] if l["entity_id"] == entity_id
                    )
                else:
                    profile, links = random.choice(list(profiles.items()))
                    link = random.choice(links)
                entities = create_split_proxies(link["entity"])
                try:
                    A_entity, B_entity = list(islice(entities, 2))
                    yield (A_entity, B_entity), True, profile
                    class_counts[True] += 1
                    pbar.update(1)
                except ValueError:
                    pass


def extract_important_fields(samples):
    field_counts = Counter(k for sample in samples for k in sample.keys())
    cutoff = np.mean(list(field_counts.values())) - np.std(list(field_counts.values()))
    fields = [field for field, count in field_counts.most_common() if count > cutoff]
    fields_to_remove = set()
    for field in fields:
        unique_values = set()
        for sample in samples:
            unique_values.add(sample.get(field))
            if len(unique_values) == 3 or (
                None not in unique_values and len(unique_values) == 2
            ):
                break
        else:
            fields_to_remove.add(field)
    fields = [field for field in fields if field not in fields_to_remove]
    return fields


def extract_data(data):
    samples = []
    features = []
    ftm_scores = []
    schemas = []
    indicies = []
    for i, (A, B) in tqdm(enumerate(data), desc="Creating features", total=len(data)):
        sample = {
            "name": compare.compare_names(A, B),
            "country": compare.compare_countries(A, B),
        }
        feature = {
            "name": (A.names, B.names),
            "country": (A.country_hints, B.country_hints),
        }
        ftm_score = (
            sample["name"] * compare.NAMES_WEIGHT
            + sample["country"] * compare.COUNTRIES_WEIGHT
        )
        try:
            schema = model.common_schema(A.schema, B.schema)
        except InvalidData:
            continue
        for name, prop in schema.properties.items():
            if name in FIELDS_BAN_LIST:
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
            feature[prop.name] = (A_values, B_values)
            sample[prop.name] = prop_score
            ftm_score += weight * prop_score
        if len(sample) == 2:
            continue
        samples.append(sample)
        features.append(feature)
        indicies.append(i)
        ftm_scores.append(ftm_score)
        schemas.append(schema)
    fields = extract_important_fields(samples)
    X = np.asarray(
        [[sample.get(field, np.NaN) for field in fields] for sample in samples]
    )
    return (
        X,
        features,
        np.asarray(ftm_scores).reshape(-1, 1),
        fields,
        schemas,
        indicies,
    )


def train_test_split(
    *arrays, test_pct=0.25, correlation_marker=None, balance_classes=None, shuffle=True
):
    M = len(arrays[0])
    print("Num samples:", M)
    arrays = (*arrays, list(range(M)))
    N = len(arrays)
    assert all(len(a) == M for a in arrays)
    cdata = defaultdict(lambda: [[] for _ in range(N)])
    for c, *items in zip(correlation_marker, *arrays):
        for i, item in enumerate(items):
            cdata[c][i].append(item)
    data = list(cdata.values())
    data.sort(key=lambda x: len(x[0]), reverse=True)
    splits = [data[0], data[1]]
    for items in data[2:]:
        n_train = len(splits[0][0])
        n_test = len(splits[1][0])
        if n_test / (n_train + n_test) < test_pct:
            s = 1
        else:
            s = 0
        for i, item in enumerate(items):
            splits[s][i].extend(item)
    if balance_classes is not None:
        for s in splits:
            n_pos = sum(balance_classes[i] for i in s[-1])
            n_neg = sum(not balance_classes[i] for i in s[-1])
            n_remove = abs(n_pos - n_neg)
            class_indicies = [
                i
                for i, j in enumerate(s[-1])
                if balance_classes[j] == bool(n_pos > n_neg)
            ]
            to_remove = list(random.sample(class_indicies, n_remove))
            print(f"Removing {n_remove} from the {n_pos > n_neg} class")
            to_remove.sort(reverse=True)
            for i in to_remove:
                for d in s:
                    d.pop(i)
    for split in splits:
        # Remove index array
        split.pop()
        if shuffle:
            new_indicies = list(range(len(split[0])))
            random.shuffle(new_indicies)
            for i in range(len(split)):
                split[i] = [split[i][j] for j in new_indicies]
    return [np.asarray(d) for train, test in zip(*splits) for d in (train, test)]
