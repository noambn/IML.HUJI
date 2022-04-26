from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        self.vars_ = np.zeros((self.classes_.size, X.shape[1]))
        for i, cl in zip(range(self.classes_.size), self.classes_):
            means.append(np.mean(X[y == cl], axis=0))
            self.vars_[i, :] = np.var(X[y == cl], axis=0, ddof=1)
        self.mu_ = np.asarray(means)
        self.vars_ = np.asarray(self.vars_)
        print("classes:", self.classes_)
        print("mu:", self.mu_)
        print("vars:", self.vars_.shape, self.vars_)

        self.pi_ = np.zeros(0)
        for i, label in zip(range(self.classes_.size), self.classes_):
            self.pi_ = np.insert(self.pi_, i, np.mean(y == label), axis=0)
        print("pi: ", self.pi_)

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
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        # raise NotImplementedError()
        m, d = X.shape
        num_classes = self.classes_.size
        ll = np.zeros((m, num_classes))
        log_pi_sum = 0
        for k in range(num_classes):
            pi_k = self.pi_[k]
            cl_count = pi_k * m
            log_pi_sum += np.log(pi_k)
            for i in range(m):
                mean = (X[i] - self.mu_[k]) ** 2
                var_divided_by_sigma = mean / self.vars_[k]
                inner_sum = np.log(self.vars_[k]) + var_divided_by_sigma
                ll[i, k] = -0.5 * np.sum(inner_sum)

        # print("check")
        md_log = m * d * np.log(2 * np.pi) / 2
        print("log_pi_sum:", log_pi_sum)
        ll = ll + log_pi_sum - md_log
        print("ll:", ll)

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
