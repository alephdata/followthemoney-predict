import itertools as IT
import json

import numpy as np
from sklearn.metrics import confusion_matrix


class DefaultNone(object):
    pass


def merge_iters(*iters):
    fillvalue = DefaultNone
    for items in IT.zip_longest(*iters, fillvalue=fillvalue):
        yield from filter(lambda i: i is not fillvalue, items)


def load_jsonl(fd):
    for line in fd:
        data = json.loads(line)
        yield data


def load_links(fd):
    for data in load_jsonl(fd):
        if data["entity"].get("schema") and data.get("decision") is not None:
            yield data


def load_collection(fd):
    yield from load_jsonl(fd)


def print_important_features(fields, scores):
    scores = np.asarray(scores)
    scores /= scores.sum()
    important_features = list(zip(scores, fields))
    important_features.sort(reverse=True)
    print(
        "Important Features:",
        ", ".join(f"{f} ({s*100:0.2f}%)" for s, f in important_features if s > 0),
    )


def print_confusion_per_schema(model, samples, targets, schemas):
    print(confusion_matrix(targets, model.predict(samples)))
    return
    schemas_uniq = set(schemas)
    for schema in schemas_uniq:
        idxs = [i for i, s in enumerate(schemas) if s == schema]
        prediction = model.predict(samples[idxs])
        print(f"Schema: {schema.name}")
        print(confusion_matrix(targets[idxs], prediction))

