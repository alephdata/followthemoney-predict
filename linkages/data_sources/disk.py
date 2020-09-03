from pathlib import Path
import warnings

import orjson

from .. import const
from .common import StreamSet, ENTITYSET_CACHE_DIR, COLLECTION_CACHE_DIR

DATAPATH = (Path(__file__).parent / "../data").relative_to(Path.cwd())


def get_data_streams(pipeline, negative_collections=const.NEGATIVE_COLLECTION_FIDS):
    if not ENTITYSET_CACHE_DIR.exists() or not COLLECTION_CACHE_DIR.exists():
        raise FileNotFoundError(
            "Disk caches are empty. Run in aleph mode with caching to fill cache."
        )

    profile_data_stream = pipeline.read_text(ENTITYSET_CACHE_DIR / "*.jsonl.gz").map(
        orjson.loads
    )

    negative_data_stream = pipeline.read_text(
        [
            COLLECTION_CACHE_DIR / f"{collection_foreign_id}.jsonl.gz"
            for collection_foreign_id in negative_collections
        ],
    ).map(orjson.loads)

    positive_data_stream = pipeline.read_text(COLLECTION_CACHE_DIR / "*.jsonl.gz").map(
        orjson.loads
    )

    return StreamSet(profile_data_stream, negative_data_stream, positive_data_stream)
