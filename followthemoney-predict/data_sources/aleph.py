import itertools as IT

from alephclient.api import AlephAPI
from tqdm import tqdm

from .common import cache_entityset, cache_collection, retry_aleph_exception

MAX_ENTITIES_PER_COLLECTION = 1_000_000
ALEPH_PARAMS = {"retries": 10}


@cache_entityset
@retry_aleph_exception
def get_entityset_items(entityset, schema=None):
    api = AlephAPI(**ALEPH_PARAMS)
    setitems = api.entitysetitems(entityset["id"], publisher=True)
    if schema:
        setitems = filter(lambda e: e["schema"] in schema, setitems)
    setitems = IT.islice(setitems, MAX_ENTITIES_PER_COLLECTION)
    yield from setitems


@cache_collection
@retry_aleph_exception
def get_entities(collection, schema):
    api = AlephAPI(**ALEPH_PARAMS)
    entities = api.stream_entities(collection, schema=schema, publisher=True)
    entities = IT.islice(entities, MAX_ENTITIES_PER_COLLECTION)
    yield from entities


def get_entitysets(set_types=None):
    api = AlephAPI(**ALEPH_PARAMS)
    entitysets = api.entitysets(set_types=set_types)
    total = entitysets.result.get("total")
    for entityset in tqdm(entitysets, desc="Reading Profiles", total=total):
        yield entityset


def get_all_collections():
    if collections is None:
        api = AlephAPI(**ALEPH_PARAMS)
    yield from api.filter_collections("*")


def get_collection_by_foreign_id(foreign_id):
    api = AlephAPI(**ALEPH_PARAMS)
    return api.get_collection_by_foreign_id(foreign_id)
