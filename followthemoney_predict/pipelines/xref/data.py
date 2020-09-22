import logging
from collections import namedtuple

from tqdm import tqdm

from . import settings

StreamSet = namedtuple("StreamSet", "profile negative positive".split(" "))


def get_data_streams(workflow, data_source, negative_collections):
    logging.debug("Getting all collections and profiles")
    collections = list(data_source.get_all_collections())
    entity_sets = list(data_source.get_entitysets(set_types="profile"))
    profile_stream = (
        workflow.from_sequence(tqdm(entity_sets, desc="Profile EntitySets"))
        .map(data_source.get_entityset_items, schema=settings.SCHEMAS)
        .flatten()
    )
    negative_stream = (
        workflow.from_sequence(tqdm(negative_collections, desc="Negative Collections"))
        .map(data_source.get_collection_by_foreign_id)
        .map(data_source.get_entities, schema=settings.SCHEMAS)
        .flatten()
    )
    positive_stream = (
        workflow.from_sequence(tqdm(collections, desc="Positive Collections"))
        .map(data_source.get_entities, schema=settings.SCHEMAS)
        .flatten()
    )
    return StreamSet(
        profile_stream,
        negative_stream,
        positive_stream,
    )
