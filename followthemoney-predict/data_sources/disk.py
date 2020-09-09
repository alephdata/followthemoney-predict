from pathlib import Path
import gzip

import orjson

from .common import ENTITYSET_CACHE_DIR, COLLECTION_CACHE_DIR

DATAPATH = (Path(__file__).parent / "../data").relative_to(Path.cwd())


def get_entityset_items(entityset, schema=None):
    with gzip.open(ENTITYSET_CACHE_DIR / "*.jsonl.gz") as fd:
        for line in fd:
            data = orjson.loads(line)
            if not schema or data["schema"] in schema:
                yield data


def get_entities(collection, schema=None):
    collection_foreign_id = collection["foreign_id"]
    with gzip.open(COLLECTION_CACHE_DIR / f"{collection_foreign_id}.jsonl.gz") as fd:
        for line in fd:
            data = orjson.loads(line)
            if not schema or data["schema"] in schema:
                yield data


def get_entitysets(set_types=None):
    for metafile in ENTITYSET_CACHE_DIR.glob("*_meta.json"):
        with open(metafile) as fd:
            meta = orjson.loads(fd)
        if meta["type"] in set_types:
            yield meta


def get_all_collections():
    for metafile in COLLECTION_CACHE_DIR.glob("*_meta.json"):
        with open(metafile) as fd:
            yield orjson.loads(fd)


def get_collection_by_foreign_id(collection_foreign_id):
    """
    TODO: Should this raise an exception or return an empty dict?
    """
    try:
        with open(COLLECTION_CACHE_DIR / f"{collection_foreign_id}_meta.json") as fd:
            return orjson.loads(fd)
    except FileNotFoundError:
        return {}
