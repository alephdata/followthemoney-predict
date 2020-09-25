import logging

import numpy as np
import pandas as pd

from ..pipeline import ftm_features_from_proxy
from .util import get_phases, xarray
from .xref_model import XrefModel


class XrefFTMModel(XrefModel):
    __model_registry = {}

    def __init_subclass__(cls):
        super().__init_subclass__()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit_parquet(self, fd):
        self.meta["fit_data"] = fd.name
        df = pd.read_parquet(fd).reset_index(drop=True)
        return self.fit(df)

    def prepair_train_test(self, df):
        source_weight = {"negative": 0.1, "positive": 0.1, "profile": 10}
        judgement_counts = dict(df.judgement.value_counts())
        judgement_weight = {
            k: 1 - v / sum(judgement_counts.values())
            for k, v in judgement_counts.items()
        }
        df["weight"] = df.apply(
            lambda row: source_weight[row.source] * judgement_weight[row.judgement],
            axis=1,
        )
        phases = get_phases(df)
        train, test = phases["train"], phases["test"]
        return train, test

    def predict(self, df):
        logging.debug(f"Creating prediction on {df.shape[0]} samples")
        X = xarray(df.features)
        return self.predict_array(X)

    def predict_array(self, X):
        return self.clf.predict_proba(X)

    def compare(self, ftm_model, A, B):
        """Mimick the followthemoney.compare:compare API"""
        schema = ftm_model.common_schema(A.schema, B.schema)
        features = ftm_features_from_proxy(A, B, schema)
        prediction = self.predict_array(features[np.newaxis])[0]
        return prediction[1]
