import numpy as np
import pandas as pd

from .. import settings


def value_or_first_list_item(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def aux_fields(sample, prefix):
    for feature in settings.FEATURE_IDXS:
        key = f"{prefix}_{feature}"
        value = value_or_first_list_item(sample.get(key, pd.NA))
        if feature != "name" and pd.notna(value):
            yield f"{feature[:2]}: {value[:6]}"


def format_prediction(sample, p):
    p *= 100
    left_nonnone = ", ".join(aux_fields(sample, "left"))
    right_nonnone = ", ".join(aux_fields(sample, "right"))

    left_name = value_or_first_list_item(sample.left_name)
    right_name = value_or_first_list_item(sample.right_name)

    return f"    [{sample.source[:3]}] {left_name} ({left_nonnone}) vs {right_name} ({right_nonnone})-> {{ F: {p[0]:0.2f}, T: {p[1]:0.2f} }}"


def get_phases(df):
    phases = {}
    for phase in df.phase.cat.categories:
        phases[phase] = (
            df.query(f"phase == '{phase}'").sample(frac=1).reset_index(drop=True)
        )
    return phases


def xarray(X):
    return np.asarray([*X])
