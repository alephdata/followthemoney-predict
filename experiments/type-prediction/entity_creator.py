from collections import Counter
from collections import namedtuple

from followthemoney import model
from followthemoney.types import registry
from followthemoney.exc import InvalidData
import fasttext

TYPE_LOOKUP = {
    f'__label__{k}': v
    for k, v in registry.groups.items()
}
TYPE_LOOKUP['__label__trash'] = None

SCHEMA_LOOKUP = {
    f'__label__{k}': v
    for k, v in model.schemata.items()
}


ModelResult = namedtuple('ModelResult', 'label confidence'.split(' '))


class EntityCreator:
    def __init__(self, type_model, entity_model, k=10):
        self.type_model = type_model
        self.entity_model = entity_model
        self.k = k
        self._schemas = 'Thing'
        self._properties = []
        self._name_type = registry.get('name')

    def add(self, data, infer_schema=True):
        type_labels, confidence = self.type_model.predict(data, k=self.k)
        data_type = [TYPE_LOOKUP[type_label] for type_label in type_labels]
        type_results = Counter(dict(zip(data_type, confidence)))
        self._properties.append((type_results, data))

        if infer_schema and self._name_type in type_results:
            prob_factor = type_results[self._name_type]
            schema_results = self.infer_schema(data, prob_factor=prob_factor)
        else:
            schema_results = None

        return type_results,  schema_results

    def infer_schema(self, name, prob_factor=1.0):
        schema_labels, confidence = self.entity_model.predict(name, k=self.k)
        confidence = [c * prob_factor for c in confidence]
        schemas = [SCHEMA_LOOKUP[schema_label] for schema_label in schema_labels]
        schema_results = Counter(dict(zip(schemas, confidence)))
        self._schemas = schema_results
        return schema_results

    def proxy(self, skip_trash=True):
        best_proxies = Counter()
        for schema, schema_confidence in self._schemas.items():
            confidence = 1.0 / schema_confidence
            proxy = model.make_entity(schema)
            schema_properties = proxy.schema.properties.values()
            try:
                for type_result, value in self._properties:
                    for type_label, type_confidence in type_result.most_common():
                        if type_label == None:
                            if skip_trash:
                                continue
                            else:
                                confidence += 1.0 / type_confidence
                                break
                        found_prop = None
                        # Find the first schema property that is in the type
                        # group we identified
                        for schema_property in schema_properties:
                            if schema_property.type == type_label:
                                found_prop = schema_property
                                break
                        if found_prop is not None:
                            proxy.add(schema_property, value)
                            confidence += 1.0 / type_confidence
                            break
                    else:
                        raise ValueError
            except ValueError:
                continue
            confidence = 1.0 / confidence
            proxy.make_id(schema.name, *proxy.properties.keys())
            best_proxies[proxy] = confidence
        return best_proxies
