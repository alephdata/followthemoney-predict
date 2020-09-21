import pickle

import git
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from followthemoney_predict.util import multi_open
from .util import format_prediction


class XrefModel:
    __model_registry = {}

    def __init__(self, *args, **kwargs):
        repo = git.Repo(search_parent_directories=True)
        self.revision = repo.head.object.hexsha[:8]
        self.version = self.version or None
        self.meta = getattr(self, "meta") or {}
        self.clf = getattr(self, "clf") or None

    def __init_subclass__(cls):
        XrefModel.__model_registry[cls.__name__] = cls

    @staticmethod
    def get_model(name):
        return XrefModel.__model_registry[name]

    def dump(self, filename, token=None, **kwargs):
        with multi_open(filename, "wb", token=token, **kwargs) as fd:
            return fd.write(self.dumps())

    def dumps(self):
        meta = self.meta or {}
        return pickle.dumps(
            {
                "version": {
                    "type": self.__class__.__name__,
                    "version": self.version,
                    "revision": self.revision,
                },
                "meta": meta,
                "model": self.clf,
            }
        )

    @classmethod
    def load(cls, filename, token=None, **kwargs):
        with multi_open(filename, "rb", token=token, **kwargs) as fd:
            return cls.loads(fd.read())

    @staticmethod
    def loads(blob):
        data = pickle.loads(blob)
        version = data["version"]
        model_cls = XrefModel.get_model(version["type"])
        model = model_cls.__new__(model_cls)

        model.clf = data["model"]
        model.meta = data["meta"]
        model.version = version["version"]
        model.revision = version["revision"]
        return model

    def better_than(self, other, metric="roc_auc"):
        try:
            return self.meta["scores"]["roc_auc"] > other.meta["scores"]["roc_auc"]
        except KeyError:
            raise ValueError("Model not fitted")

    def __repr__(self):
        return f"<{self.__class__.__name__}:{self.revision}:{self.version}>"

    def describe(self, df):
        y = df.judgement
        y_predict_proba = self.predict(df)
        y_predict = y_predict_proba[:, 0] < y_predict_proba[:, 1]

        sources_indexes = {s: df.source == s for s in set(df.source.cat.categories)}
        sources_accuracy = {
            s: accuracy_score(y[idxs], y_predict[idxs])
            for s, idxs in sources_indexes.items()
        }

        accuracy = accuracy_score(y, y_predict)
        roc_auc = roc_auc_score(y, y_predict_proba[:, 1])
        confusion = 100 * confusion_matrix(y, y_predict, normalize="true")
        pi = sources_indexes["profile"]
        confusion_profile = 100 * confusion_matrix(
            y[pi], y_predict[pi], normalize="true"
        )

        scores = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "confusion": confusion,
            "confusion_profile": confusion_profile,
            "source_accuracy": sources_accuracy,
        }

        print(f"Model Accuracy: {accuracy*100:0.4f}%")
        print(f"Model ROC AUC: {roc_auc}")

        print("Per source accuracy:")
        for s, a in sources_accuracy.items():
            print(f"\t{s}: {a*100:0.2f}%")

        print("Model Confusion Matrix")
        print(" \tN' \t P'")
        print(f"N\t{confusion[0,0]:0.2f}\t{confusion[0, 1]:0.2f}")
        print(f"P\t{confusion[1,0]:0.2f}\t{confusion[1, 1]:0.2f}")

        print("Model Confusion Matrix on Profiles")
        print(" \tN' \t P'")
        print(f"N\t{confusion_profile[0,0]:0.2f}\t{confusion_profile[0, 1]:0.2f}")
        print(f"P\t{confusion_profile[1,0]:0.2f}\t{confusion_profile[1, 1]:0.2f}")

        print("Accuracy per threshold")
        N = y_predict_proba.shape[0]
        for t in np.arange(0, 1, 0.05):
            correct = sum((y_predict_proba[:, 1] > t) == y)
            print(
                f"\t[{t:0.3f}] Accuracy: {correct / N * 100:0.2f}% ({correct} correct)"
            )

        self.describe_predictions(df, y_predict_proba)
        return scores

    @staticmethod
    def describe_predictions(df, y_predict_proba):
        certain_negative_indicies = np.argsort(y_predict_proba[:, 0])
        certain_positive_indicies = np.argsort(y_predict_proba[:, 1])
        uncertain_indicies = np.argsort(
            np.abs(y_predict_proba[:, 0] - y_predict_proba[:, 1])
        )
        print("Certain Positives")
        for i in reversed(certain_positive_indicies[-5:]):
            if y_predict_proba[i][1] > 0.5:
                print(format_prediction(df.iloc[i], y_predict_proba[i]))

        print("Certain Negatives")
        for i in reversed(certain_negative_indicies[-5:]):
            if y_predict_proba[i][0] > 0.5:
                print(format_prediction(df.iloc[i], y_predict_proba[i]))

        print("Uncertain Predictions")
        for i in uncertain_indicies[:10]:
            print(format_prediction(df.iloc[i], y_predict_proba[i]))
