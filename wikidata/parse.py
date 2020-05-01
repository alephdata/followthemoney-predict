import json
import click
import requests
from pprint import pprint
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import SKOS, RDFS

SCHEMA = Namespace("http://schema.org/")
PROP = Namespace("http://www.wikidata.org/prop/direct/")

ENTITY = Namespace('http://www.wikidata.org/entity/')
SPECIAL = 'https://www.wikidata.org/wiki/Special:EntityData/'

SPARQL = 'https://query.wikidata.org/sparql'
SPARQL_TYPE = 'application/sparql-results+json;charset=utf-8'

# Filter out non-latin alphabets that we don't support in
# other parts of the stack anyway and that will not well
# survive transliteration:
SKIP_LANGUAGES = set([
    'ast', 'ja', 'zh', 'zh-hant', 'zh-hk', 'ko',
    'id', 'th', 'ia', 'gan', 'si', 'zh-classical',
    'nv', 'or', 'arc', 'cu', 'lo', 'iu', 'got',
    'chr', 'cr', 'bug', 'dz', 'gom', 'tcy', 'dty',
    'sat', 'nqo', 'mnw', 'pi', 'shn', 'ii',  'hi',
    'bn', 'te', 'mr', 'gu', 'ta', 'zh-yue', 'my',
    'gu', 'kn', 'ml', 'ne', 'pa', 'as', 'new',
    'bpy', 'sa', 'mai', 'wuu', 'km', 'am', 'dv',
])

PERSON = str(ENTITY['Q5'])
MAPPINGS = {
    PROP.P31: 'type',
    PROP.P21: 'gender',
    PROP.P27: 'nationality',
    # PROP.P735: 'firstName',
    # PROP.P734: 'lastName',
    PROP.P569: 'birthDate',
    # PROP.P19: 'birthPlace',
    PROP.P1477: 'alias',
    SCHEMA.name: 'name',
    RDFS.label: 'name',
    SKOS.prefLabel: 'name',
    SKOS.altLabel: 'weakAlias'
}
LABELS = {}


def query_labels(query):
    res = requests.get(SPARQL, params={'query': query},
                       headers={'Accept': SPARQL_TYPE})
    data = res.json()
    for row in data.get('results', {}).get('bindings', []):
        node = row.get('node', {}).get('value')
        label = row.get('label', {}).get('value')
        if node is not None and label is not None:
            if node not in LABELS:
                LABELS[node] = label


def fetch_labels():
    # Anything with an ISO country code:
    q = "SELECT ?node ?label WHERE { ?node wdt:P297 ?label . }"
    query_labels(q)
    clazzes = [
        'Q6256',  # country
        'Q3624078',  # sovereign state
        'Q3024240',  # historical state
        'Q4369513',  # sexes
        'Q48264',  # genders
    ]
    for clazz in clazzes:
        q = "SELECT ?node ?label WHERE { ?node wdt:P31 wd:%s . ?node rdfs:label ?label . FILTER langMatches(lang(?label),'EN') }"  # noqa
        q = q % clazz
        query_labels(q)
    # pprint(LABELS)


def parse_triples(fh, size=1000):
    line_no = 0
    while True:
        line = fh.readline()
        line_no += 1
        if not line:
            break
        # if line_no % 1000 == 0:
        #     print("LINE", line_no)
        try:
            graph = Graph()
            graph.parse(data=line, format='nt')
            yield from graph
        except Exception:
            pass


@click.command()
@click.option('-i', '--input', type=click.File('r'), default='-')  # noqa
@click.option('-o', '--output', type=click.File('w'), default='-')  # noqa
def transform(input, output):
    prev = None
    entity = {}
    for (s, p, o) in parse_triples(input):
        if s != prev and prev is not None:
            if PERSON in entity.pop('type', []):
                data = {}
                data['wikidataId'] = [str(prev)]
                for key, values in entity.items():
                    data[key] = []
                    for value in values:
                        if key == 'name' and len(value) < 4:
                            continue
                        if value.startswith('https://www.wikidata.org'):
                            continue
                        if key == 'birthDate':
                            value = value[:10]
                        data[key].append(value)
                if len(data.get('name', [])):
                    output.write(json.dumps(data))
                    output.write('\n')
            entity = {}
        prev = s
        field = MAPPINGS.get(p)
        if field is not None:
            entity.setdefault(field, set())
            if isinstance(o, Literal):
                if o.language in SKIP_LANGUAGES:
                    continue
            value = str(o)
            value = LABELS.get(value, value)
            entity[field].add(value)


if __name__ == '__main__':
    fetch_labels()
    transform()
