from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # raise NotImplementedError()
        plus = 1
        minus = -1
        sign = plus
        best_f = 0
        best_thr = 0
        min_err = np.inf
        for f in range(X.shape[1]):
            sign = plus
            thr, thr_error = self._find_threshold(X[:, f], y, plus)
            thr_m, thr_error_m = self._find_threshold(X[:, f], y, minus)
            if thr_error > thr_error_m:
                thr, thr_error = thr_m, thr_error_m
                sign = minus
            if min_err > thr_error:
                min_err = thr_error
                best_f = f
                best_thr = thr

        self.j_ = best_f
        self.threshold_ = best_thr
        self.sign_ = sign


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        # raise NotImplementedError()
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)



    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # raise NotImplementedError()
        # thr_err = 1.0
        # thr = 0.0
        # # checks all the potential thresholds
        # for t in values:
        #     b = np.where(values >= t, sign, -sign)
        #     # should I seperate it like in the algorithm?
        #     g = misclassification_error(y_true=labels, y_pred=b)
        #     if thr_err > g:
        #         thr_err = g
        #         thr = t
        # thr_err = np.sum(labels[np.sign(labels) == sign])
        # thr = np.concatenate([[-np.inf], (values[1:] + values[:-1]) / 2, [np.inf]])
        # losses = np.append(thr_err, thr_err - np.cumsum(labels * sign))
        # min_loss = np.argmin(losses)
        # return thr[min_loss], losses[min_loss]

        sorted = np.argsort(values)
        values, labels = values[sorted], labels[sorted]
        thr_err = np.sum(np.abs(labels[np.sign(labels) == sign]))
        thr = np.concatenate(
            [[-np.inf], (values[1:] + values[:-1]) / 2, [np.inf]])
        losses = np.append(thr_err, thr_err - np.cumsum(labels * sign))
        minimal_loss = np.argmin(losses)
        return thr[minimal_loss], losses[minimal_loss]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        # raise NotImplementedError()
        y_pred = self.predict(X)
        return misclassification_error(y_true=y, y_pred=y_pred)