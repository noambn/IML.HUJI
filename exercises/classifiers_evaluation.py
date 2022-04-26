from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        # raise NotImplementedError()
        X, Y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        # raise NotImplementedError()

        # def callback()
        def callback(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(X, Y))

        p_estimator = Perceptron(callback=callback)
        p_estimator.fit(X, Y)
        print(losses)

        # Plot figure of loss as function of fitting iteration
        # raise NotImplementedError()
        fitting_iteration = np.linspace(1, 1000, num=1000)
        fig = go.Figure((go.Scatter(x=fitting_iteration, y=losses,
                                mode="lines", name="Fitting iteration",
                                marker=dict(color="green"))))
        fig.update_layout(
            title_text=f"(Question 1) Loss as function of fitting iteration - {n}",
            xaxis_title='Loss', yaxis_title='Fitting iteration',
            title_font_size=30, width=1200, height=700)
        # fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        # raise NotImplementedError()
        X, Y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        # raise NotImplementedError()
        lda_estimator = LDA()
        lda_estimator.fit(X, Y)
        lda_y_prad = lda_estimator.predict(X)

        gnb_estimator = GaussianNaiveBayes()
        gnb_estimator.fit(X, Y)
        gnb_y_pred = gnb_estimator.predict(X)


        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        # raise NotImplementedError()
        gnb_accuracy = accuracy(Y, gnb_y_pred)
        lda_accuracy = accuracy(Y, lda_y_prad)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f"Gaussian naive bayes with accuracy of {gnb_accuracy}", f"Linear discriminant analysis with accuracy of {lda_accuracy}"))
        # add the GNB trace
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1],
                                mode="markers",
                                line=dict(dash="dash"),
                                marker=dict(color="green", opacity=.7)), row=1, col=1)

        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1],
                                 mode="markers",
                                 line=dict(dash="dash"),
                                 marker=dict(color="green", opacity=.7)),
                      row=1, col=1)

        fig.update_layout(height=500, width=700,
                          title_text="(Bayes Classifiers - Question 1)")

        fig.show()

        # Add traces for data-points setting symbols and colors
        raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
