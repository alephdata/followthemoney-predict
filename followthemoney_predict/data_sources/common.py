import gzip
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import orjson
from alephclient.api import AlephException


class DataSource:
    def __init__(self, **settings):
        self.settings = settings

    def get_collection_entities_by_foreign_id(self, foreign_id, schema=None):
        collection = self.get_collection_by_foreign_id(foreign_id)
        yield from self.get_entities(collection, schema)

    @staticmethod
    def _entityset_cache(cache_dir):
        return Path(cache_dir) / "entityset_cache/"

    @staticmethod
    def _collection_cache(cache_dir):
        return Path(cache_dir) / "collection_cache/"


@contextmanager
def open_tmpwriter(filename, *args, suffix=".tmp", open=open, **kwargs):
    filename_tmp = Path(str(filename) + suffix)
    with open(filename_tmp, *args, **kwargs) as fd:
        yield fd
    filename_tmp.rename(filename)


def cache_entityset(fxn, cache_dir):
    return cache_with_meta(
        cache_dir=DataSource._entityset_cache(cache_dir),
        key_fxn=lambda meta: meta["id"],
    )(fxn)


def cache_collection(fxn, cache_dir):
    return cache_with_meta(
        cache_dir=DataSource._collection_cache(cache_dir),
        key_fxn=lambda meta: meta["foreign_id"],
    )(fxn)


def cache_with_meta(cache_dir, key_fxn=None):
    if key_fxn is None:
        key_fxn = lambda meta: meta["id"]  # NOQA

    def cache(fxn):
        def _(meta, *args, **kwargs):
            if not meta:
                return fxn(meta, *args, **kwargs)

            key = key_fxn(meta)
            cache_meta = cache_dir / f"{key}_meta.json"
            cache_data = cache_dir / f"{key}.jsonl.gz"
            if cache_meta.exists():
                with open(cache_meta) as fd:
                    meta_disk = orjson.loads(fd.read())
                cache_update_at = datetime.fromisoformat(meta_disk["updated_at"])
                meta_update_at = datetime.fromisoformat(meta["updated_at"])
                if cache_data.exists() and meta_update_at >= cache_update_at:
                    logging.debug(f"Getting cached entries for {cache_data}")
                    with gzip.open(cache_data) as fd:
                        # fd = tqdm(fd, desc=f"Reading from cache: {key}", leave=False)
                        for line in fd:
                            yield orjson.loads(line)
                    return
            cache_meta.parent.mkdir(parents=True, exist_ok=True)
            cache_data.parent.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Creating cached entries for {cache_data}")
            with open_tmpwriter(cache_meta, "wb+") as fd:
                fd.write(orjson.dumps(meta))
                with open_tmpwriter(cache_data, "wb+", open=gzip.open) as fd:
                    for item in fxn(meta, *args, **kwargs):
                        fd.write(orjson.dumps(item))
                        fd.write(b"\n")
                        yield item

        return _

    return cache


def retry_aleph_exception(fxn):
    def _(*args, **kwargs):
        n_skip = 0
        n_tries = 0
        while True:
            try:
                n_items = 0
                for item in fxn(*args, **kwargs):
                    n_items += 1
                    if n_items > n_skip:
                        yield item
                break
            except AlephException as e:
                n_skip = n_items
                n_tries += 1
                logging.exception(
                    f"Aleph Exception... trying again: Attempt {n_tries}: {e}"
                )
                time.sleep(min(2 ** n_tries, 60))

    return _
