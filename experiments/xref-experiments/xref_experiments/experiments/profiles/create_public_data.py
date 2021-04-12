import json
from pathlib import Path

from alephclient.api import AlephAPI


data_path = Path('../../../data/profiles/')


def filter_public(item, collection_fids):
    return not item or item['collection']['foreign_id'] in collection_fids


if __name__ == "__main__":
    api = AlephAPI('https://aleph.occrp.org/', 'XXX')
    collections = list(api.filter_collections('*'))
    collection_fids = {c['foreign_id'] for c in collections}
    data = []
    for c in collection_fids:
        try:
            with open(data_path / f'{c}_profiles.json') as fd:
                for line in fd:
                    d = json.loads(line)
                    if filter_public(d['entity'], collection_fids) and filter_public(d['compared_to_entity'], collection_fids):
                        data.append(d)
        except FileNotFoundError:
            pass
    for d in data:
        print(json.dumps(d))

