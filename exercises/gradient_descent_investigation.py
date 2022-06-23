import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection.cross_validate import cross_validate
from IMLearn.metrics import loss_functions
from utils import *
from sklearn.metrics import roc_curve, auc
pio.renderers.default = "browser"

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration
    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm
    values: List[np.ndarray]
        Recorded objective values
    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights_lst = []
    norms = []

    def callback(solver, weights, val, grad, t, eta, delta):
        values.append(val)
        weights_lst.append(weights)
        norms.append(delta)

    return callback, values, weights_lst, norms


def compare_fixed_learning_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        etas: Tuple[float] = (1, .1, .01, .001)):
    for eta in etas:
        for l_rate, l_rate_str in {L1: 'L1', L2: 'L2'}.items():
            callback, values, weights_lst, norms = get_gd_state_recorder_callback()
            GradientDescent(FixedLR(eta), callback=callback).fit(l_rate(init), None, None)
            fig1 = plot_descent_path(module=l_rate, descent_path=np.asarray(weights_lst),
                                    title=f'(eta {eta}, {l_rate_str} module)')
            gd_iterations = len(norms)
            fig2 = go.Figure(
                go.Scatter(x=np.linspace(0, gd_iterations - 1, gd_iterations), y=norms,
                           mode="lines+markers")) \
                .update_layout(title=f'Convergence rate (eta {eta}, {l_rate_str})', title_font_size=15, width=1200, height=700)
            if eta == .01:
                fig1.show()
                fig2.show()
                print(
                    f'Minimal loss ({l_rate_str} module, eta {eta}): {np.min(values)}.')


def compare_exponential_decay_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        eta: float = .1,
        gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = go.Figure().update_layout(
        title='Convergence rate of decay rates')
    for gamma in gammas:
        callback, values, weights_lst, norms = get_gd_state_recorder_callback()
        GradientDescent(ExponentialLR(eta, gamma), callback=callback).fit(L1(init), None, None)
        gd_iterations = len(norms)
        fig.add_trace(
            go.Scatter(x=np.linspace(0, gd_iterations - 1, gd_iterations), y=norms,
                       mode="lines+markers", name=f'Decay rate: {gamma}')).update_layout(title_font_size=15, width=1200, height=700)

    # Plot algorithm's convergence for the different values of gamma
    fig.show()

    # Plot descent path for gamma=0.95
    for l_rate, l_rate_str in {L1: "L1", L2: "L2"}.items():
        callback, values, weights_lst, norms = get_gd_state_recorder_callback()
        GradientDescent(ExponentialLR(eta, gammas[1]), callback=callback).fit(l_rate(init), None, None)
        plot_descent_path(l_rate, np.asarray(weights_lst),
                          f'with eta {eta} on {l_rate_str} module.').show()
        print(f'Minimal norm ({l_rate_str} module): {np.min(norms)}.')


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion
    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset
    train_portion: float, default=0.8
        Portion of dataset to use as a training set
    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set
    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples
    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set
    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    logistic_reg = LogisticRegression()
    logistic_reg.fit(np.asarray(X_train), np.asarray(y_train))
    fpr, tpr, thresholds = roc_curve(np.asarray(y_train), logistic_reg.predict_proba(np.asarray(X_train)))
    best_alpha = np.round(thresholds[np.argmax(tpr-fpr)], 2)
    print("best alpha: ", best_alpha)

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines',
                         text=thresholds, name="", showlegend=False,
                         marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(
                title=r"$\text{True Positive Rate (TPR)}$"))).update_layout(title_font_size=15, width=1200, height=700).show()



    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambads = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for lr in ["l1", "l2"]:
        train_scores = []
        test_scores = []
        for lam in lambads:
            logistic_reg = LogisticRegression(penalty=lr, lam=lam)
            train_score, test_score = cross_validate(logistic_reg, np.asarray(X_train), np.asarray(y_train), loss_functions.misclassification_error)
            train_scores.append(train_score)
            test_scores.append(test_score)
        best_lam = lambads[np.argmin(test_scores)]
        reg_of_best_lam = LogisticRegression(penalty=lr, lam=best_lam).fit(np.asarray(X_train), np.asarray(y_train))
        test_error = reg_of_best_lam.loss(np.asarray(X_test), np.asarray(y_test))
        print("best lambda:", best_lam, "test error: ", test_error)


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()