from functools import singledispatch

from followthemoney import model
from followthemoney.proxy import EntityProxy
from followthemoney.schema import Schema


@singledispatch
def filter_schema(schema):
    raise ValueError


@filter_schema.register(dict)
def _(entity):
    schema = model.get(entity["schema"])
    return filter_schema(schema)


@filter_schema.register(EntityProxy)
def _(proxy):
    return filter_schema(proxy.schema)


@filter_schema.register(Schema)
def _(schema):
    return schema.is_a(model.get("Thing")) and not schema.is_a(model.get("Document"))
