import itertools as IT

from alephclient.api import AlephAPI
from tqdm import tqdm

from .. import const
from .common import StreamSet, cache_entityset, cache_collection, retry_aleph_exception

MAX_ENTITIES_PER_COLLECTION = 1_000_000
ALEPH_PARAMS = {"retries": 10}


@cache_entityset
@retry_aleph_exception
def get_entityset(entityset):
    api = AlephAPI(**ALEPH_PARAMS)
    setitems = IT.islice(
        api.entitysetitems(entityset["id"], publisher=True), MAX_ENTITIES_PER_COLLECTION
    )
    yield from setitems


@cache_collection
@retry_aleph_exception
def get_entities(collection):
    api = AlephAPI(**ALEPH_PARAMS)
    entities = api.stream_entities(collection, schema=const.SCHEMAS, publisher=True)
    entities = IT.islice(entities, MAX_ENTITIES_PER_COLLECTION)
    yield from entities


def get_profiles():
    api = AlephAPI(**ALEPH_PARAMS)
    entitysets = api.entitysets(set_types="profile")
    total = entitysets.result.get("total")
    for entityset in tqdm(entitysets, desc="Reading Profiles", total=total):
        yield entityset


def entities_negative(negative_collections):
    api = AlephAPI(**ALEPH_PARAMS)
    for fid in tqdm(negative_collections, desc="Reading Negatives"):
        collection = api.get_collection_by_foreign_id(fid)
        yield collection


def entities_positive():
    api = AlephAPI(**ALEPH_PARAMS)
    collections = api.filter_collections("*")
    for collection in tqdm(collections, desc="Reading Positives"):
        yield collection


def get_data_streams(pipeline, negative_collections=const.NEGATIVE_COLLECTION_FIDS):
    profile_stream = (
        pipeline.from_sequence(get_profiles()).map(get_entityset, pipeline).flatten()
    )
    negative_stream = (
        pipeline.from_sequence(entities_negative(negative_collections))
        .map(get_entities)
        .flatten()
    )
    positive_stream = (
        pipeline.from_sequence(entities_positive()).map(get_entities).flatten()
    )
    return StreamSet(profile_stream, negative_stream, positive_stream,)
