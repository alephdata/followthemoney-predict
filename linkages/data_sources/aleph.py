import itertools as IT

from alephclient.api import AlephAPI
from tqdm import tqdm

from .. import const
from .common import StreamSet, cache_entityset, cache_collection, retry_aleph_exception

MAX_ENTITIES_PER_COLLECTION = 1_000_000
ALEPH_PARAMS = {"retries": 10}


def get_profiles():
    api = AlephAPI(**ALEPH_PARAMS)
    entitysets = api.entitysets(set_types="profile")
    total = entitysets.result.get("total")
    for entityset in tqdm(entitysets, desc="Reading Profiles", total=total):
        yield from get_entityset(entityset)


@cache_entityset
@retry_aleph_exception
def get_entityset(entityset):
    api = AlephAPI(**ALEPH_PARAMS)
    setitems = IT.islice(
        api.entitysetitems(entityset["id"], publisher=True), MAX_ENTITIES_PER_COLLECTION
    )
    setitems = tqdm(
        setitems, desc=f"Reading EntitySet items: {entityset['id']}", leave=False,
    )
    yield from setitems


@cache_collection
@retry_aleph_exception
def get_entities(collection):
    api = AlephAPI(**ALEPH_PARAMS)
    entities = api.stream_entities(collection, schema=const.SCHEMAS, publisher=True)
    entities = IT.islice(entities, MAX_ENTITIES_PER_COLLECTION)
    entities = tqdm(
        entities,
        desc=f"Reading Collection Entities: {collection['foreign_id']}",
        leave=False,
    )
    yield from entities


def entities_negative(negative_collections):
    api = AlephAPI(**ALEPH_PARAMS)
    for fid in tqdm(negative_collections, desc="Reading Negatives"):
        collection = api.get_collection_by_foreign_id(fid)
        yield from get_entities(collection)


def entities_positive():
    api = AlephAPI(**ALEPH_PARAMS)
    collections = api.filter_collections("*")
    for collection in tqdm(collections, desc="Reading Positives"):
        yield from get_entities(collection)


def get_data_streams(pipeline, negative_collections=const.NEGATIVE_COLLECTION_FIDS):
    profile_stream = get_profiles()
    negative_stream = entities_negative(negative_collections)
    positive_stream = entities_positive()
    return StreamSet(
        pipeline.from_sequence(profile_stream),
        pipeline.from_sequence(negative_stream),
        pipeline.from_sequence(positive_stream),
    )
