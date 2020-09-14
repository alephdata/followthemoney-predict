import logging

import numpy as np
import pandas as pd
from followthemoney import model as ftm_model

from .xref_base_model import XrefBaseModel
from .util import get_phases, xarray
from ..pipeline import ftm_features_from_proxy


class XrefFTMModel(XrefBaseModel):
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
        WEIGHTS = {"negative": 0.1, "positive": 0.1, "profile": 10}
        df["weight"] = df.source.apply(WEIGHTS.__getitem__)
        phases = get_phases(df)
        train, test = phases["train"], phases["test"]
        return train, test

    def predict(self, df):
        logging.debug(f"Creating prediction on {df.shape[0]} samples")
        X = xarray(df.features)
        return self.clf.predict_proba(X)

    def compare(self, A, B):
        """Mimick the followthemoney.compare:compare API"""
        schema = ftm_model.common_schema(A.schema, B.schema)
        features = ftm_features_from_proxy(A, B, schema)
        prediction = self.predict_proba(features[np.newaxis])
        return prediction[1]
