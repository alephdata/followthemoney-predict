from pathlib import Path
import ujson

from .. import const
from .common import StreamSet


DATAPATH = (Path(__file__).parent / '../data').relative_to(Path.cwd())

def load_json(item):
    blob, path = item
    data = ujson.loads(blob)
    data["path"] = path
    return data


def get_data_streams(pipeline, negative_collections=const.NEGATIVE_COLLECTION_FIDS):
    profile_data_stream = pipeline.read_text(
        DATAPATH / "profiles/*", include_path=True
    ).map(load_json)

    negative_data_stream = pipeline.read_text(
        [
            DATAPATH / f".entities/raw-latest/legal_entity-{collection.replace(' ', '_')}.json"
            for collection in negative_collections
        ],
        include_path=True,
    ).map(load_json)

    positive_data_stream = pipeline.read_text(
        DATAPATH / "entities/legal_entities-multi_name/*.json", include_path=True
    ).map(load_json)

    return StreamSet(profile_data_stream, negative_data_stream, positive_data_stream)
