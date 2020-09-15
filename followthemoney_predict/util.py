import gzip
import itertools as IT
import json
import logging
import os
from types import GeneratorType

import click
import gcsfs
import numpy as np
from sklearn.metrics import confusion_matrix


class DefaultNone(object):
    pass


class FileAnyType(click.File):
    def _ensure_call(self, fxn, ctx):
        if ctx is not None:
            ctx.call_on_close(click.utils.safecall(fxn))

    def convert(self, value, param, ctx):
        is_bytes = isinstance(value, bytes)
        if not is_bytes and not isinstance(value, str):
            return super().convert(value, param, ctx)
        if is_bytes:
            value = value.decode("utf8")

        try:
            fd = multi_open(value, self.mode, use_file=False)
            self._ensure_call(fd.close, ctx)
            return super().convert(fd, param, ctx)
        except gcsfs.core.HttpError as e:
            return self.fail(
                f"Could not open GCS file: {value}: {e}",
                param,
                ctx,
            )
        except ValueError:
            pass
        return super().convert(value, param, ctx)


def multi_open(
    filename, mode, use_gcs=True, use_file=True, use_gzip=True, token=None, **kwargs
):
    if use_gcs and (filename.startswith("gcs://") or filename.startswith("gc://")):
        token = token or os.environ.get("FTM_PREDICT_GCS_TOKEN")
        logging.debug(f"Using GCSFS to open file: {filename}:{mode}")
        fs = gcsfs.GCSFileSystem(token=token)
        return fs.open(filename, mode=mode, **kwargs)
    elif use_gzip and (filename.endswith(".gz") or filename.endswith(".gzip")):
        logging.debug(f"Using GZIP to open file: {filename}:{mode}")
        return gzip.open(filename, mode, **kwargs)
    elif use_file:
        return open(filename, mode, **kwargs)
    raise ValueError(f"Unable to open file: {filename}:{mode}")


def unify_map(fxn, workflow):
    if workflow.IS_DASK:

        def _(*args, **kwargs):
            result = fxn(*args, **kwargs)
            if isinstance(result, GeneratorType):
                return list(result)
            return result

        return _
    return fxn


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
