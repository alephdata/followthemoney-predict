import logging

import xgboost as xgb

from .util import xarray
from .xref_ftm_model import XrefFTMModel


class XrefXGBoost(XrefFTMModel):
    version = "1.0"

    def __init__(self):
        self.meta = {
            "init_args": {
                "colsample_bylevel": 1.0,
                "colsample_bytree": 0.9,
                "gamma": 1.2,
                "learning_rate": 0.2,
                "max_depth": 18,
                "min_child_weight": 0.6,
                "n_estimators": 120,
                "objective": "binary:logistic",
                "reg_lambda": 0.2,
                "subsample": 0.8,
                "n_jobs": -1,
            }
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
