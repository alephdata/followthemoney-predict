import itertools as IT

from alephclient.api import AlephAPI
from tqdm import tqdm

from .. import const
from ..util import merge_iters
from .common import StreamSet


def get_profiles(api):
    entityset_query = api.entitysets(set_types="profile")
    entitysets = (es["id"] for es in entityset_query)
    total = entityset_query.result["total"]
    for entityset in tqdm(entitysets, desc="Reading Profiles", total=total):
        setitems = api.entitysetitems(entityset, publisher=True)
        yield from zip(setitems, IT.cycle([entityset]))


def get_collection_entities(api, collection):
    fid = collection["foreign_id"]
    entities = api.stream_entities(collection, schema=const.SCHEMAS, publisher=True)
    entities = zip(entities, IT.cycle([fid]))
    yield from tqdm(entities, desc="Reading Collection Entities", leave=False)


def _entities_negative(api, negative_collections):
    for fid in tqdm(negative_collections, desc="Reading Negatives"):
        collection = api.get_collection_by_foreign_id(fid)
        yield from get_collection_entities(api, collection)


def _entities_positive(api, negative_collections):
    collections = api.filter_collections("*")
    total = collections.result["total"]
    for collection in tqdm(collections, desc="Reading Positives", total=total):
        fid = collection["foreign_id"]
        if fid in negative_collections:
            continue
        yield from get_collection_entities(api, collection)


def entities_neg_pos_multiplex(api, negative_collections):
    entities_neg, entities_pos = IT.tee(_entities_negative(api, negative_collections))
    entities_pos_chain = merge_iters(
        _entities_positive(api, negative_collections), entities_pos,
    )
    return entities_neg, entities_pos_chain


def map_path(item):
    data, path = item
    data["path"] = path
    return data


def get_data_streams(pipeline, negative_collections=const.NEGATIVE_COLLECTION_FIDS):
    api = AlephAPI()
    profile_stream = get_profiles(api)
    negative_stream, positive_stream = entities_neg_pos_multiplex(
        api, negative_collections
    )
    return StreamSet(
        pipeline.from_sequence(profile_stream).map(map_path),
        pipeline.from_sequence(negative_stream).map(map_path),
        pipeline.from_sequence(positive_stream).map(map_path),
    )
