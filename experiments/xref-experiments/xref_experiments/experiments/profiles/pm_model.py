from itertools import combinations_with_replacement

import pymc3 as pm
import numpy as np
import arviz as ar
import pandas as pd
import seaborn as sns
from sklearn import metrics

from utils import TARGETS, HAS_TARGETS


class PMModel:
    can_sample = True

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
        if not self.can_sample:
            return
        with self.model:
            # self._trace = pm.sample(1000, tune=1000, progressbar=True)
            self._trace = pm.sample(
                2_500, tune=1000, progressbar=True, return_inferencedata=False
            )
            return self._trace

    def summarize(self, ax=None):
        if not self.can_sample:
            return
        assert hasattr(self, "_trace")
        with self.model:
            print(ar.summary(self._trace))
        return self.traceplot(ax)

    def traceplot(self, ax=None):
        if not self.can_sample:
            return
        with self.model:
            return ar.plot_trace(self._trace, axes=ax)

    def precision_recall_accuracy_curve(self, data, ax=None):
        estimate = self.evaluate(data, mle=True)
        precision, recall, threshold = metrics.precision_recall_curve(
            data.y, estimate, sample_weight=data.weight
        )
        T = np.arange(0, 1, 0.05)
        accuracy = [
            metrics.accuracy_score(data.y, estimate > t, sample_weight=data.weight)
            for t in T
        ]
        max_accuracy = max(accuracy)
        sns.lineplot(x=threshold, y=recall[:-1], label="recall", ax=ax)
        sns.lineplot(x=threshold, y=precision[:-1], label="precision", ax=ax)
        ax = sns.lineplot(
            x=T, y=accuracy, label=f"accuracy ({max_accuracy*100:0.1f}%)", ax=ax
        )
        ax.hlines(y=max_accuracy, xmin=0, xmax=1, alpha=0.5)
        ax.set_xlim(0, 1)
        return ax

    def estimate_distribution(self, data, ax=None):
        data["estimate"] = self.evaluate(data, mle=True)
        fg = sns.kdeplot(
            data=data, x="estimate", hue="judgement", cumulative=True, ax=ax
        )
        return fg


class RandomModel(PMModel):
    can_sample = False

    def evaluate(self, data, mle=True):
        shape = [data.shape[0]]
        if not mle:
            shape += [
                100,
            ]
        return np.random.rand(*shape)

    def setup(self, data):
        return {}


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


class GLMBernoulli2E(PMModel):
    def __init__(self, *args, targets=None, targets_2e=None, **kwargs):
        if targets_2e is None:
            self.targets_2e = list(combinations_with_replacement(targets, 2))
        else:
            self.targets_2e = targets_2e
        super().__init__(*args, targets=targets, **kwargs)

    def evaluate(self, data, mle=True):
        weights = {
            "coef": self._trace.get_values("coef"),
            "coef_2e": self._trace.get_values("coef_2e"),
            "intercept": self._trace.get_values("intercept"),
        }
        d = data[self.targets].fillna(0)
        d2 = np.asarray([d[t1] * d[t2] for t1, t2 in self.targets_2e]).T
        if mle:
            score = (
                np.inner(weights["coef"].mean(axis=0), d.values)
                + np.inner(weights["coef_2e"].mean(axis=0), d2)
                + weights["intercept"].mean(axis=0)
            )
        else:
            score = (
                np.inner(weights["coef"], d.values)
                + np.inner(weights["coef_2e"], d2)
                + weights["intercept"][:, np.newaxis]
            ).T
        return 1.0 / (1.0 + np.exp(-score))

    def setup(self, data):
        data[self.targets] = data[self.targets].fillna(0)
        data_2e = np.asarray([data[t1] * data[t2] for t1, t2 in self.targets_2e]).T
        with self.model:
            weights = {
                "coef": pm.Normal("coef", 0, 5, shape=len(self.targets)),
                "coef_2e": pm.Normal("coef_2e", 0, 5, shape=len(self.targets_2e)),
                "intercept": pm.Normal("intercept", 0, 5),
            }
            score = (
                (weights["coef"] * data[self.targets].values).sum(axis=-1)
                + (weights["coef_2e"] * data_2e).sum(axis=-1)
                + weights["intercept"]
            )
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


class TargetWeightedBeta(_TargetWeightedUnitDist):
    def distribution(self, name, shape):
        return pm.Beta(name, 2, 5, shape=shape)


class TargetWeightedDirichlet(_TargetWeightedUnitDist):
    def distribution(self, name, shape):
        return pm.Dirichlet(name, a=np.asarray([1] * shape))


if __name__ == "__main__":
    df = pd.read_pickle("../../../data/profiles/profiles_processed.pkl")

    d = df.sample(frac=0.75)
    glm_bernoulli = GLMBernoulli(d)
    glm_bernoulli.sample()
