from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"
import sys
from utils import *
from scipy.stats import norm


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    # raise NotImplementedError()
    mu1, sigma1 = 10, 1
    s = np.random.normal(mu1, sigma1, 1000)
    estimator1 = UnivariateGaussian()
    estimator1.fit(s)
    print((estimator1.mu_, estimator1.var_))

    # Question 2 - Empirically showing sample mean is consistent
    # raise NotImplementedError()
    X = np.linspace(10, 1000, 100)
    Y = []
    for i in X:
        si = s[:int(i)]
        estimator = UnivariateGaussian()
        estimator.fit(si)
        Y.append(abs(estimator.mu_ - mu1))
    print(Y)
    fig = make_subplots(rows=1, cols=1).add_traces([go.Scatter(x=X, y=Y, mode='lines', marker=dict(color="black"), showlegend=False)],rows=[1], cols=[1])
    fig.update_xaxes(title_text="Number of samples")
    fig.update_yaxes(title_text="Loss")
    fig.update_layout(
        title_text=r"$\text{(question 2) Loss as function of sample size}$")
    # fig.show()
    # Question 3 - Plotting Empirical PDF of fitted model
    # raise NotImplementedError()
    # X = s
    Y = estimator1.pdf(s)
    fig2 = make_subplots(rows=1, cols=1).add_traces([go.Scatter(x=s, y=Y, mode='markers', marker=dict(color="black"), showlegend=False)],rows=[1], cols=[1])
    fig2.update_xaxes(title_text="sample value")
    fig2.update_yaxes(title_text="PDF calculation")
    fig2.update_layout(
        title_text=r"$\text{(question 3) probability density function by each sample}$", title_font_size=30)
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    # raise NotImplementedError()
    mu = [0, 0, 4, 0]
    sigma = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    s = np.random.multivariate_normal(mu, sigma, 1000)
    estimator = MultivariateGaussian()
    estimator.fit(s)
    # print((estimator.mu_, estimator.cov_))
    print(estimator.pdf(s))

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    test_multivariate_gaussian()
