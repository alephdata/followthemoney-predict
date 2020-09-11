import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

import xgboost as xgb

from . import settings


def model_predict(model, df):
    logging.debug(f"Creating prediction on {df.shape[0]} samples")
    X = xarray(df.features)
    return model.predict_proba(X)


def describe_model(model, df):
    y = df.judgement
    y_predict_proba = model_predict(model, df)
    y_predict = y_predict_proba[:, 0] < y_predict_proba[:, 1]

    sources_indexes = {s: df.source == s for s in set(df.source.cat.categories)}
    sources_accuracy = {
        s: accuracy_score(y[idxs], y_predict[idxs])
        for s, idxs in sources_indexes.items()
    }

    accuracy = accuracy_score(y, y_predict)
    roc_auc = roc_auc_score(y, y_predict)
    confusion = 100 * confusion_matrix(y, y_predict, normalize="true")
    pi = sources_indexes["profile"]
    confusion_profile = 100 * confusion_matrix(y[pi], y_predict[pi], normalize="true")

    meta = {
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

    describe_model_predictions(df, y_predict_proba)
    return meta


def describe_model_predictions(df, y_predict_proba):
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


def fit_xgboost(df):
    WEIGHTS = {"negative": 0.1, "positive": 0.1, "profile": 10}
    df["weight"] = df.source.apply(WEIGHTS.__getitem__)
    phases = get_phases(df)
    train, test = phases["train"], phases["test"]

    clf = xgb.XGBClassifier(
        reg_lambda=5e-4, gamma=1, max_depth=4, learning_rate=0.1, n_jobs=-1
    )
    clf.fit(
        xarray(train.features),
        train.judgement,
        sample_weight=train.weight,
        early_stopping_rounds=25,
        eval_metric="auc",
        eval_set=[(xarray(test.features), test.judgement)],
        sample_weight_eval_set=[test.weight],
    )
    meta = describe_model(clf, test)
    return clf, meta


def fit_linear(df):
    WEIGHTS = {"negative": 0.1, "positive": 0.1, "profile": 10}
    df["weight"] = df.source.apply(WEIGHTS.__getitem__)
    phases = get_phases(df)
    train, test = phases["train"], phases["test"]

    clf = LogisticRegression(max_iter=5000, n_jobs=-1, verbose=False)
    clf.fit(
        xarray(train.features),
        train.judgement,
        sample_weight=train.weight,
    )
    meta = describe_model(clf, test)
    return clf, meta
