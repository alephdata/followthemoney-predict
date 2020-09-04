import time
from collections import namedtuple
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import gzip

import orjson
from alephclient.api import AlephException
from tqdm import tqdm

COLLECTION_CACHE_DIR = (
    Path(__file__).parent / Path("../data/collection_cache/")
).relative_to(Path.cwd())
ENTITYSET_CACHE_DIR = (
    Path(__file__).parent / Path("../data/entityset_cache/")
).relative_to(Path.cwd())

StreamSet = namedtuple("StreamSet", "profile negative positive".split(" "))


@contextmanager
def open_tmpwriter(filename, *args, suffix=".tmp", open=open, **kwargs):
    filename_tmp = Path(str(filename) + suffix)
    with open(filename_tmp, *args, **kwargs) as fd:
        yield fd
    filename_tmp.rename(filename)


def cache_entityset(fxn):
    return cache_with_meta(
        cache_dir=ENTITYSET_CACHE_DIR, key_fxn=lambda meta: meta["id"]
    )(fxn)


def cache_collection(fxn):
    return cache_with_meta(
        cache_dir=COLLECTION_CACHE_DIR, key_fxn=lambda meta: meta["foreign_id"]
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
                    with gzip.open(cache_data) as fd:
                        # fd = tqdm(fd, desc=f"Reading from cache: {key}", leave=False)
                        for line in fd:
                            yield orjson.loads(line)
                    return
            cache_meta.parent.mkdir(parents=True, exist_ok=True)
            cache_data.parent.mkdir(parents=True, exist_ok=True)
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
                print(f"Aleph Exception... trying again: Attempt {n_tries}: {e}")
                time.sleep(min(2 ** n_tries, 60))

    return _
