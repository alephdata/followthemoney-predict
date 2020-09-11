import gzip
from pathlib import Path

import orjson

from .common import DataSource


class DiskSource(DataSource):
    def __init__(self, cache_dir, **settings):
        self.cache_dir = Path(cache_dir)
        self.entityset_cache = self._entityset_cache(self.cache_dir)
        self.collection_cache = self._collection_cache(self.cache_dir)
        super().__init__(**settings)

    def get_entityset_items(self, entityset, schema=None):
        _id = entityset["id"]
        with gzip.open(self.entityset_cache / f"{_id}.jsonl.gz") as fd:
            for line in fd:
                data = orjson.loads(line)
                if not schema or data.get("entity", {}).get("schema") in schema:
                    yield data

    def get_entities(self, collection, schema=None):
        collection_foreign_id = collection["foreign_id"]
        with gzip.open(
            self.collection_cache / f"{collection_foreign_id}.jsonl.gz"
        ) as fd:
            for line in fd:
                data = orjson.loads(line)
                if not schema or data["schema"] in schema:
                    yield data

    def get_entitysets(self, set_types=None):
        for metafile in self.entityset_cache.glob("*_meta.json"):
            with open(metafile) as fd:
                meta = orjson.loads(fd.read())
            if set_types is None or meta["type"] in set_types:
                yield meta

    def get_all_collections(self):
        for metafile in self.collection_cache.glob("*_meta.json"):
            with open(metafile) as fd:
                yield orjson.loads(fd.read())

    def get_collection_by_foreign_id(self, collection_foreign_id):
        """
        TODO: Should this raise an exception or return an empty dict?
        """
        try:
            with open(
                self.collection_cache / f"{collection_foreign_id}_meta.json"
            ) as fd:
                return orjson.loads(fd.read())
        except FileNotFoundError:
            return {}
