import pymc3 as pm
import numpy as np
import arviz as ar
import pandas as pd
import seaborn as sns
from sklearn import metrics

from utils import TARGETS, HAS_TARGETS


class PMModel:
    def __init__(self, data, targets=TARGETS, has_targets=HAS_TARGETS):
        self.model = pm.Model()
        self.targets = targets
        self.has_targets = has_targets
        self.weights = self.setup(data)

    @property
    def has_sampled(self):
        return hasattr(self, "_trace")

    def evaluate(self, data):
        raise NotImplementedError

    def setup(self, data):
        raise NotImplementedError

    def weighted_bernoili_logli(self, data, mu):
        with self.model:
            logp = data.weight.values * pm.Bernoulli.dist(p=mu).logp(data.y.values)
            error = pm.Potential("error", logp)
        return error

    def sample(self):
        with self.model:
            # self._trace = pm.sample(1000, tune=1000, progressbar=True)
            self._trace = pm.sample(
                1000, tune=1000, progressbar=True, return_inferencedata=False
            )
            return self._trace

    def summarize(self, ax=None):
        assert hasattr(self, "_trace")
        with self.model:
            print(ar.summary(self._trace))
        return self.traceplot(ax)

    def traceplot(self, ax=None):
        with self.model:
            return ar.plot_trace(self._trace, axes=ax)

    def precision_recall_curve(self, data, ax=None):
        estimate = self.evaluate(data, mle=True)
        precision, recall, threshold = metrics.precision_recall_curve(
            data.y, estimate, sample_weight=data.weight
        )
        sns.lineplot(x=threshold, y=recall[:-1], label="recall", ax=ax)
        ax = sns.lineplot(x=threshold, y=precision[:-1], label="precision", ax=ax)
        return ax

    def estimate_distribution(self, data, ax=None):
        data["estimate"] = self.evaluate(data, mle=True)
        fg = sns.kdeplot(
            data=data, x="estimate", hue="judgement", cumulative=True, ax=ax
        )
        return fg


class GLMBernoulli(PMModel):
    def evaluate(self, data, mle=True):
        weights = {
            "coef": self._trace.get_values("coef"),
            "intercept": self._trace.get_values("intercept"),
        }
        d = data[self.targets].fillna(0).values
        if mle:
            score = np.inner(weights["coef"].mean(axis=0), d) + weights[
                "intercept"
            ].mean(axis=0)
        else:
            score = np.inner(weights["coef"], d) + weights["intercept"][np.newaxis, :]
        return 1.0 / (1.0 + np.exp(-score))

    def setup(self, data):
        data[self.targets] = data[self.targets].fillna(0)
        with self.model:
            weights = {
                "coef": pm.Normal("coef", 0, 5, shape=len(self.targets)),
                "intercept": pm.Normal("intercept", 0, 5),
            }
            score = (weights["coef"] * data[self.targets].values).sum(
                axis=-1
            ) + weights["intercept"]
            mu = pm.math.invlogit(score)
            weights["error"] = self.weighted_bernoili_logli(data, mu)
            self._weights = weights
            return weights


class _TargetWeightedUnitDist(PMModel):
    def evaluate(self, data, mle=True):
        coef = self._trace.get_values("coef")
        data[self.targets] = data[self.targets].fillna(0)
        data[self.has_targets] = data[self.has_targets].fillna(False).astype("int")
        if mle:
            coef = coef.mean(axis=0)
        score = np.inner(coef, data[self.targets])
        norm = np.inner(coef, data[self.has_targets])
        return np.nan_to_num(score / norm, nan=0)

    def setup(self, data):
        data[self.has_targets] = data[self.has_targets].fillna(False).astype("int")
        with self.model:
            weights = {
                "coef": self.distribution("coef", shape=len(self.targets)),
            }

            score = (weights["coef"] * data[self.targets].values).sum(axis=-1)
            norm = (weights["coef"] * data[self.has_targets].values).sum(axis=-1)

            mu = score / norm
            weights["error"] = self.weighted_bernoili_logli(data, mu)
            self._weights = weights
            return weights


class _FieldWeightedUnitDist(PMModel):
    def evaluate(self, data, mle=True):
        coef = self._trace.get_values("coef")
        partial = self._trace.get_values("partial")
        data[self.targets] = data[self.targets].fillna(0)
        data[self.has_targets] = data[self.has_targets].fillna(False).astype("int")
        if mle:
            coef = coef.mean(axis=0)
            partial = partial.mean(axis=0)
        score = np.inner(coef, data[self.targets])
        norm = len(self.targets) * (data["pct_full"] + partial * data["pct_partial"])
        return np.nan_to_num(score / norm, nan=0)

    def setup(self, data):
        data[self.has_targets] = data[self.has_targets].fillna(False).astype("int")
        with self.model:
            weights = {
                "coef": self.distribution("coef", shape=len(self.targets)),
                "partial": self.distribution("partial", shape=1),
            }

            score = (weights["coef"] * data[self.targets].values).sum(axis=-1)
            norm = len(self.targets) * (
                data["pct_full"].values + weights["partial"] * data["pct_partial"]
            )

            mu = score / norm
            weights["error"] = self.weighted_bernoili_logli(data, mu)
            self._weights = weights
            return weights


class TargetWeightedBeta(_TargetWeightedUnitDist):
    def distribution(self, name, shape):
        return pm.Beta(name, 2, 5, shape=shape)


class TargetWeightedDirichlet(_TargetWeightedUnitDist):
    def distribution(self, name, shape):
        return pm.Dirichlet(name, a=np.asarray([1] * shape))


class FieldWeightedBeta(_FieldWeightedUnitDist):
    def distribution(self, name, shape):
        return pm.Beta(name, 2, 5, shape=shape)


class FieldWeightedDirichlet(_FieldWeightedUnitDist):
    def distribution(self, name, shape):
        return pm.Dirichlet(name, a=np.asarray([1] * shape))


if __name__ == "__main__":
    df = pd.read_pickle("../../../data/profiles/profiles_processed.pkl")

    d = df.sample(frac=0.75)
    glm_bernoulli = GLMBernoulli(d)
    glm_bernoulli.sample()
