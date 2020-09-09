import os
import sys
import csv
from pprint import pprint  # noqa
from followthemoney import model
from followthemoney.types import registry
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

COLLECTIONS = 'aleph-collection-v1'
# Only export data from (semi-) open collections:
OPEN = ['1', '2', '3']

INCLUDES = ['schema', 'properties.*']

INDEX = [
    'aleph-entity-person-v1',
    'aleph-entity-company-v1',
    # 'aleph-entity-publicbody-v1',
    # 'aleph-entity-legalentity-v1',
]

es_url = os.environ.get('ELASTICSEARCH_URL')
es = Elasticsearch(es_url, timeout=90)

# Find all the collection_ids from the semi-open groups:
collection_ids = []
query = {
    'query': {'terms': {'team_id': OPEN}},
    '_source': {'excludes': '*'}
}
for res in scan(es, index=COLLECTIONS, query=query):
    collection_ids.append(res.get('_id'))

# Iterate all their entities:
args = {
    'query': {
        'query': {'terms': {'collection_id': collection_ids}},
        '_source': {'includes': INCLUDES}
    },
    'index': ','.join(INDEX),
    'raise_on_error': False,
    'scroll': '5m',
}
writer = csv.writer(sys.stdout, dialect=csv.unix_dialect)
# writer.writerow(['schema', 'property', 'value'])
for idx, res in enumerate(scan(es, **args)):
    if idx % 10000 == 0:
        print("Dumped %d..." % idx, file=sys.stderr)
    proxy = model.get_proxy(res.get('_source', {}))
    for prop, value in proxy.itervalues():
        if prop.type != registry.name:
            continue
        writer.writerow([proxy.schema.name, prop.name, value])
