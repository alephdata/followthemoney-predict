import itertools as IT

from alephclient.api import AlephAPI
from tqdm import tqdm

from .common import DataSource, cache_collection, cache_entityset, retry_aleph_exception

ALEPH_PARAMS = {"retries": 10}


class AlephSource(DataSource):
    def __init__(
        self,
        aleph_params=None,
        max_entities_per_collection=1_000_000,
        cache_dir="./cache/",
        **settings,
    ):
        self.aleph_params = aleph_params or ALEPH_PARAMS
        self.max_entities_per_collection = max_entities_per_collection
        self.cache_dir = cache_dir
        super().__init__(**settings)

        self.get_entities = cache_collection(self.get_entities, self.cache_dir)
        self.get_entityset_items = cache_entityset(
            self.get_entityset_items, self.cache_dir
        )

    @retry_aleph_exception
    def get_entityset_items(self, entityset, schema=None):
        api = self._aleph_api()
        setitems = api.entitysetitems(entityset["id"], publisher=True)
        if schema:
            setitems = filter(lambda e: e.get("schema") in schema, setitems)
        setitems = IT.islice(setitems, self.max_entities_per_collection)
        yield from setitems

    @retry_aleph_exception
    def get_entities(self, collection, schema=None):
        if collection.get("restricted"):
            return
        api = self._aleph_api()
        entities = api.stream_entities(collection, schema=schema, publisher=True)
        entities = IT.islice(entities, self.max_entities_per_collection)
        yield from entities

    def get_entity(self, entity_id):
        api = self._aleph_api()
        return api.get_entity(entity_id, publisher=True)

    def get_entitysets(self, set_types=None):
        api = self._aleph_api()
        entitysets = api.entitysets(set_types=set_types)
        total = entitysets.result.get("total")
        for entityset in tqdm(entitysets, desc="Reading Profiles", total=total):
            yield entityset

    def get_all_collections(self):
        api = self._aleph_api()
        yield from api.filter_collections("*")

    def get_collection_by_foreign_id(self, foreign_id):
        api = self._aleph_api()
        return api.get_collection_by_foreign_id(foreign_id)

    def _aleph_api(self):
        return AlephAPI(**self.aleph_params)
