import logging

import xgboost as xgb

from .util import xarray
from .xref_ftm_model import XrefFTMModel


class XrefXGBoost(XrefFTMModel):
    version = "0.2"

    def __init__(self):
        self.meta = {
            "init_args": dict(
                reg_lambda=5e-4,
                gamma=1,
                max_depth=4,
                learning_rate=0.1,
                n_jobs=-1,
                subsample=0.8,
            )
        }
        self.clf = xgb.XGBClassifier(**self.meta["init_args"])
        super().__init__()

    def fit(self, df):
        logging.debug(f"Training xgboost model on dataframe with shape: {df.shape}")
        train, test = self.prepair_train_test(df)
        fit_args = self.meta["fit_args"] = dict(
            early_stopping_rounds=10, eval_metric="auc"
        )
        self.clf.fit(
            xarray(train.features),
            train.judgement,
            sample_weight=train.weight,
            eval_set=[(xarray(test.features), test.judgement)],
            sample_weight_eval_set=[test.weight],
            **fit_args,
        )
        scores = self.describe(test)
        self.meta["scores"] = scores
        return self
