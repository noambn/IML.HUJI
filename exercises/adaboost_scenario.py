import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    # raise NotImplementedError()
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)

    losses_train = []
    losses_test = []

    for i in range(1, n_learners):
        losses_train.append(adaboost.partial_loss(train_X, train_y, i))
        losses_test.append(adaboost.partial_loss(test_X, test_y, i))
    X = np.arange(0, n_learners + 1)
    fig = go.Figure([
        go.Scatter(x=X, y=losses_train,
                   name="training losses"),
        go.Scatter(x=X, y=losses_test,
                   name="test losses")
    ])
    fig.update_layout(
        title=f"(Q1) 'Train and test errors by number of learners' (noise={noise})",
        title_font_size=15, width=1200, height=700).update_xaxes(
        title='Number of Learners').update_yaxes(title='Errors')
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    # raise NotImplementedError()
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[rf"$\textbf{{{t} Weak Learners}}$" for
                                        t in T],
                        horizontal_spacing=0.04, vertical_spacing=.08)
    for i, t in enumerate(T):
        fig.add_traces(
            [
                decision_surface(lambda X: adaboost.partial_predict(X, t),
                                 lims[0], lims[1],
                                 showscale=False),
                go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                           showlegend=False,
                           marker=dict(color=test_y,
                                       colorscale=[custom[0], custom[-1]],
                                       line=dict(color="black", width=1)))
            ],
            rows=(i // 2) + 1, cols=(i % 2) + 1
        )
    fig.update_layout(
        title=rf"$\textbf{{(Q2) Decision boundaries by number of week learners (noise={noise})}}$",
        margin=dict(t=100), title_font_size=15, width=1200,
        height=700)  # .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    # raise NotImplementedError()
    optimal_learners_number = None
    best_value = None
    for t in range(1, n_learners):
        loss = adaboost.partial_loss(test_X, test_y, t)
        if not best_value or loss < best_value:
            optimal_learners_number, best_value = t, loss
    fig = go.Figure(
        [
            decision_surface(
                lambda X: adaboost.partial_predict(X, optimal_learners_number),
                lims[0], lims[1],
                showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                       showlegend=False,
                       marker=dict(color=test_y,
                                   colorscale=[custom[0], custom[-1]],
                                   line=dict(color="black", width=1)))
        ]
    )
    acc = accuracy(test_y,
                   adaboost.partial_predict(test_X, optimal_learners_number))
    fig.update_layout(
        title=f"(Q3) Decision boundaries of optimal ensemble ({optimal_learners_number}  learners) with accuracy of {acc} (noise={noise})",
        title_font_size=15, width=1200, height=700)
    fig.show()

    # Question 4: Decision surface with weighted samples
    # raise NotImplementedError()
    fig = go.Figure(
        [
            decision_surface(
                lambda X: adaboost.partial_predict(X, adaboost.iterations_),
                lims[0], lims[1],
                showscale=False),
            go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                       showlegend=False,
                       marker=dict(color=train_y,
                                   colorscale=[custom[0], custom[-1]],
                                   size=adaboost.D_ / np.max(
                                       adaboost.D_) * 10))
        ],
    )
    fig.update_layout(
        title=f"(Q4) Decision boundaries with weighted dots of training set (noise={noise})",
        title_font_size=15, width=1200, height=700)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
