import logging

import numpy as np
import pandas as pd

from ..pipeline import ftm_features_from_proxy
from .util import xarray
from .xref_model import XrefModel


class XrefFTMModel(XrefModel):
    def __init_subclass__(cls):
        super().__init_subclass__()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit_parquet(self, fd):
        self.meta["fit_data"] = fd.name
        df = pd.read_parquet(fd).reset_index(drop=True)
        return self.fit(df)

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

    def compare_batch(self, ftm_model, A, Bs):
        features = []
        for B in Bs:
            schema = ftm_model.common_schema(A.schema, B.schema)
            feature = ftm_features_from_proxy(A, B, schema)
            features.append(feature)
        return self.predict_array(np.asarray(features))
