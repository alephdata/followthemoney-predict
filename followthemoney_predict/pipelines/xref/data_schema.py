from pandas import CategoricalDtype, StringDtype

from . import settings


def create_schema(phases, sources):
    dataframe_fields_types = {
        "name": StringDtype(),
        "schema": CategoricalDtype(settings.SCHEMAS),
        "collection_id": StringDtype(),
        "id": StringDtype(),
        "country": StringDtype(),
        "address": StringDtype(),
        "registrationNumber": StringDtype(),
        "alias": StringDtype(),
        "status": StringDtype(),
        "classification": StringDtype(),
        "gender": StringDtype(),
        "firstName": StringDtype(),
        "lastName": StringDtype(),
        "birthPlace": StringDtype(),
        "birthDate": StringDtype(),
        "idNumber": StringDtype(),
        "motherName": StringDtype(),
        "nationality": StringDtype(),
    }
    dataframe_meta = {
        f"{which}_{c}": t
        for which in ("left", "right")
        for c, t in dataframe_fields_types.items()
    }
    dataframe_meta["judgement"] = bool
    dataframe_meta["source"] = CategoricalDtype(sources)
    dataframe_meta["phase"] = CategoricalDtype(phases)
    dataframe_meta["features"] = object
    dataframe_meta["schema"] = StringDtype()
    return dataframe_meta
