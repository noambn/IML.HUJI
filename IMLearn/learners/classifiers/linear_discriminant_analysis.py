from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # raise NotImplementedError()
        means = []
        self.classes_ = np.unique(y)
        for cl in self.classes_:
            means.append(np.mean(X[y == cl], axis=0))
        self.mu_ = means
        f_num = X.shape[1]
        self.cov_ = np.zeros((f_num, f_num))
        for cl, mean in zip(self.classes_, self.mu_):
            cov = np.zeros((f_num, f_num))
            for sample in X[y == cl]:
                sample, mean = sample.reshape(2, 1), mean.reshape(2, 1)
                cov += (sample - mean) @ (sample - mean).transpose()
            self.cov_ += cov
        self.cov_ /= X.shape[0] - self.classes_.size
        self._cov_inv = inv(self.cov_)

        self.pi_ = np.zeros(0)
        for i, cl in zip(range(self.classes_.size), self.classes_):
            self.pi_ = np.insert(self.pi_, i, np.mean(y == cl), axis=0)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # raise NotImplementedError()
        return np.argmax(self.likelihood(X), axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        # raise NotImplementedError()
        d = X.shape[1]
        ll = np.zeros((X.shape[0], self.classes_.size))
        for i in range(self.classes_.size):
            cur_pi, cur_x_mu = self.pi_[i], X - self.mu_[i]
            # exponent = -0.5 * cur_x_mu.transpose() @ self._cov_inv @ cur_x_mu
            mahalanobis = np.einsum("bi,ij,bj->b", cur_x_mu, self._cov_inv, cur_x_mu)
            exponent = -0.5 * mahalanobis
            ll[:, i] = np.exp(exponent) * cur_pi / np.sqrt((2 * np.pi)**d * det(self.cov_))
        print("likelihood:", ll)
        return ll

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
        from ...metrics import misclassification_error
        # raise NotImplementedError()
        y_pred = self.predict(X)
        return misclassification_error(y_true=y, y_pred=y_pred)
