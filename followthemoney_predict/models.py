import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline

import util
import xgboost as xgb


def class_weights(targets):
    N = len(targets)
    s = sum(targets)
    return [1 - s / N, s / N]


def fit_xgboost(sampler):
    weights = class_weights(sampler.y_train)
    sample_weights = [weights[int(y)] for y in sampler.y_train]
    clf = xgb.XGBClassifier(max_depth=2, n_estimators=75, reg_lambda=1e-5)
    clf.fit(
        sampler.X_train,
        sampler.y_train,
        sample_weight=sample_weights,
        early_stopping_rounds=10,
        eval_metric="auc",
        eval_set=[(sampler.X_test, sampler.y_test)],
        verbose=False,
    )
    predict = clf.predict(sampler.X_test)
    predict_proba = clf.predict_proba(sampler.X_test)
    accuracy = accuracy_score(sampler.y_test, predict)
    roc = roc_auc_score(sampler.y_test, predict_proba[:, 1])
    print("XGBoost Model")
    print(f"Model Accuracy: {accuracy}")
    print(f"Model ROC/AUC: {roc}")
    util.print_important_features(sampler.fields, clf.feature_importances_)
    util.print_confusion_per_schema(
        clf, sampler.X_test, sampler.y_test, sampler.schemas
    )
    print("-" * 10)
    return clf


def fit_logit(sampler):
    imp = SimpleImputer(missing_values=np.nan, strategy="mean").fit(sampler.X_train)
    sampler.X_train_imp = imp.transform(sampler.X_train)
    sampler.X_test_imp = imp.transform(sampler.X_test)

    weights = class_weights(sampler.y_train)
    clf = LogisticRegression(class_weight=weights).fit(
        sampler.X_train_imp, sampler.y_train
    )
    predict = clf.predict(sampler.X_test_imp)
    predict_proba = clf.predict_proba(sampler.X_test_imp)
    accuracy = accuracy_score(sampler.y_test, predict)
    roc = roc_auc_score(sampler.y_test, predict_proba[:, 1])
    print("Logistic Regression on sample X")
    print(f"Model Accuracy: {accuracy}")
    print(f"Model ROC/AUC: {roc}")
    util.print_important_features(sampler.fields, clf.coef_[0])
    util.print_confusion_per_schema(
        clf, sampler.X_test_imp, sampler.y_test, sampler.schemas
    )
    print("-" * 10)
    return make_pipeline(imp, clf)


def fit_ftm(sampler):
    weights = class_weights(sampler.y_train)
    clf = LogisticRegression(class_weight=weights).fit(
        sampler.ftm_train, sampler.y_train
    )
    predict = clf.predict(sampler.ftm_test)
    predict_proba = clf.predict_proba(sampler.ftm_test)
    accuracy = accuracy_score(sampler.y_test, predict)
    roc = roc_auc_score(sampler.y_test, predict_proba[:, 1])
    print("Logistic Regression on FTM Score")
    print(f"Model Accuracy: {accuracy}")
    print(f"Model ROC/AUC: {roc}")
    util.print_confusion_per_schema(
        clf, sampler.ftm_test, sampler.y_test, sampler.schemas
    )
    print("-" * 10)
    return clf
