from collections import namedtuple

from . import settings

StreamSet = namedtuple("StreamSet", "profile negative positive".split(" "))


def get_data_streams(workflow, data_source, negative_collections):
    profile_stream = (
        workflow.from_sequence(data_source.get_entitysets(set_types='profile')).map(data_source.get_entityset_items, schema=settings.SCHEMAS).flatten()
    )
    negative_stream = (
        workflow.from_sequence(negative_collections)
        .map(data_source.get_collection_by_foreign_id)
        .map(data_source.get_entities, schema=settings.SCHEMAS)
        .flatten()
    )
    positive_stream = (
        workflow.from_sequence(data_source.get_all_collections()).map(data_source.get_entities, schema=settings.SCHEMAS).flatten()
    )
    return StreamSet(profile_stream, negative_stream, positive_stream,)
