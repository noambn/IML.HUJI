from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator
from IMLearn.metrics import accuracy

def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    # X = X.flatten()
    # y = y.flatten()
    kf_X = np.array_split(X, cv, axis=0)
    kf_y = np.array_split(y, cv, axis=0)

    train_scores = []
    validation_score = []
    for fold in range(cv):
        cur_fold = kf_X[fold]
        cur_fold_y = kf_y[fold]
        X_wo_fold = np.concatenate(kf_X[:fold] + kf_X[fold + 1:])
        y_wo_fold = np.concatenate(kf_y[:fold] + kf_y[fold + 1:])
        estimator.fit(X_wo_fold, y_wo_fold)
        y_pred = estimator.predict(X_wo_fold)
        y_pred_fold = estimator.predict(cur_fold)
        train_scores.append(scoring(y_wo_fold, y_pred))
        validation_score.append(scoring(cur_fold_y, y_pred_fold))

    return np.mean(train_scores), np.mean(validation_score)