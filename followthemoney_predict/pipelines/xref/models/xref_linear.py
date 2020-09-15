import logging

from sklearn.linear_model import LogisticRegression

from .util import xarray
from .xref_ftm_model import XrefFTMModel


class XrefLinear(XrefFTMModel):
    version = "0.1"

    def __init__(self):
        self.meta = {"init_args": dict(max_iter=5000, n_jobs=-1, verbose=False)}
        self.clf = LogisticRegression(**self.meta["init_args"])
        super().__init__()

    def fit(self, df):
        logging.debug(f"Training linear model on dataframe with shape: {df.shape}")
        train, test = self.prepair_train_test(df)
        self.clf.fit(
            xarray(train.features),
            train.judgement,
            sample_weight=train.weight,
        )
        scores = self.describe(test)
        self.meta["scores"] = scores
        return self
