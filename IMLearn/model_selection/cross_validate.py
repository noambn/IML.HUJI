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
    # raise NotImplementedError()
    # k_foldes = KFold(n_splits=cv)
    # k_foldes.get_n_splits(X)
    #
    # for train_index in k_foldes.split(X):
    #     X, y = X[train_index], y[train_index]

    # m = y.size
    # shuffled_inds = np.arange(m)
    # np.random.shuffle(shuffled_inds)
    # X_shuffled, y_shuffled = X.astype('float64'), y.astype('float64')
    # kf_X = np.array_split(X_shuffled, 5, axis=0)
    # kf_y = np.array_split(y_shuffled, 5, axis=0)
    # kf_X = np.array_split(X, cv, axis=0)
    # kf_y = np.array_split(y, cv, axis=0)
    #
    # # for param in range(k):  # what is k?
    # X_wo_fold = np.concatenate(kf_X[1:])
    # y_wo_fold = np.concatenate(kf_y[1:])
    # train_scores = []
    # validation_score = []
    # for fold in range(cv):
    #     cur_fold = kf_X[fold]
    #     cur_fold_y = kf_y[fold]
    #     if len(kf_y[fold+1:]) == 0:
    #         X_wo_fold = np.concatenate(kf_X[:-1])
    #         y_wo_fold = np.concatenate(kf_y[:-1])
    #     elif len(kf_X[:fold]) != 0:
    #         X_wo_fold1, X_wo_fold2  = np.concatenate(kf_X[:fold]), np.concatenate(kf_X[fold+1:])
    #         X_wo_fold = np.concatenate((X_wo_fold1, X_wo_fold2))
    #         y_wo_fold1, y_wo_fold2 = np.concatenate(kf_y[:fold]), np.concatenate(kf_y[fold+1:])
    #         y_wo_fold = np.concatenate((y_wo_fold1, y_wo_fold2))
    #     h_i = estimator.fit(X_wo_fold.flatten(), y_wo_fold)
    #     y_pred_test = h_i.predict(cur_fold.flatten())
    #     y_pred_train = h_i.predict(X_wo_fold.flatten())
    #     cur_train_score = scoring(y_wo_fold, y_pred_train)
    #     train_scores.append(cur_train_score)
    #     cur_validation_score = scoring(cur_fold_y, y_pred_test)
    #     validation_score.append(cur_validation_score)
    #
    # return np.mean(train_scores), np.mean(validation_score)

    X = X.flatten()
    y = y.flatten()
    kf_X = np.array_split(X, cv, axis=0)
    kf_y = np.array_split(y, cv, axis=0)

    # for param in range(k):  # what is k?
    X_wo_fold = np.concatenate(kf_X[1:])
    y_wo_fold = np.concatenate(kf_y[1:])
    train_scores = []
    validation_score = []
    for fold in range(cv):
        cur_fold = kf_X[fold]
        cur_fold_y = kf_y[fold]
        if len(kf_y[fold + 1:]) == 0:
            X_wo_fold = np.concatenate(kf_X[:-1])
            y_wo_fold = np.concatenate(kf_y[:-1])
        elif len(kf_X[:fold]) != 0:
            X_wo_fold1, X_wo_fold2 = np.concatenate(
                kf_X[:fold]), np.concatenate(kf_X[fold + 1:])
            X_wo_fold = np.concatenate((X_wo_fold1, X_wo_fold2))
            y_wo_fold1, y_wo_fold2 = np.concatenate(
                kf_y[:fold]), np.concatenate(kf_y[fold + 1:])
            y_wo_fold = np.concatenate((y_wo_fold1, y_wo_fold2))
        h_i = estimator.fit(X_wo_fold, y_wo_fold)
        y_pred_test = h_i.predict(cur_fold)
        y_pred_train = h_i.predict(X_wo_fold)
        cur_train_score = scoring(y_wo_fold, y_pred_train)
        train_scores.append(cur_train_score)
        cur_validation_score = scoring(cur_fold_y, y_pred_test)
        validation_score.append(cur_validation_score)

    return np.mean(train_scores), np.mean(validation_score)
