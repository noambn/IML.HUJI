from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_poli_dataset(n_samples, noise):
    m = n_samples
    eps, x = np.random.normal(0, noise, size=m), np.linspace(-1.2, 2, m)
    y_wo_noise = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    y = y_wo_noise + eps
    train_X, train_y, test_X, test_y = split_train_test(
        pd.DataFrame(x), pd.Series(y), 2 / 3)

    return x, y_wo_noise, np.asarray(train_X).flatten(), np.asarray(train_y), np.asarray(test_X).flatten(), np.asarray(test_y)


def generate_diabetes_dataset(n_samples):
    x, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = split_train_test(
        pd.DataFrame(x), pd.Series(y), n_samples / y.size)
    #
    return np.asarray(train_X), np.asarray(train_y), np.asarray(test_X), np.asarray(test_y)


def question_1(X, y_wo_noise, train_X, train_y, test_X, test_y, n_samples, noise):
    fig = go.Figure([go.Scatter(x=X, y=y_wo_noise, mode='markers+lines',
                                name='True model'),
                     go.Scatter(x=train_X, y=train_y, mode='markers',
                                name='Train set'),
                     go.Scatter(x=test_X, y=test_y, mode='markers',
                                name='Test set')])
    fig.update_layout(
        title=f"(Q1) True model and train and test sets of noise {noise} and dataset size of {n_samples}",
        title_font_size=15, width=1200, height=700)

    fig.show()

def question_2(train_X, train_y, n_samples, noise):
    training_errors = []
    validation_errors = []
    degs = np.arange(11)
    for deg in degs:
        train_error, valid_error = cross_validate(PolynomialFitting(deg),
                                              train_X, train_y,
                                              scoring=mean_square_error)
        training_errors.append(train_error)
        validation_errors.append(valid_error)

    best_deg = np.argmin(validation_errors)  # retuns the index of the min value

    fig = go.Figure([go.Scatter(x=degs, y=training_errors, mode='lines+markers', name='Average training error'),
                     go.Scatter(x=degs, y=validation_errors, mode='lines+markers', name='Average validation error')])
    fig.update_layout(
        title=f"(Q2) Error of 5-fold cross validation by degree of noise {noise} and dataset size of {n_samples}",
        title_font_size=15, width=1200, height=700)

    fig.show()

    return best_deg, validation_errors[best_deg]

def question_3(train_X, train_y, test_X, test_y, best_k, best_k_error, n_samples, noise):
    polinomial_model_best_k = PolynomialFitting(best_k).fit(train_X, train_y)
    test_error = polinomial_model_best_k.loss(test_X, test_y)
    print(f"(Q3) Noise of {noise}, dataset size of {n_samples}")
    print(f"k of lowest validation error: {best_k}")
    print(f"Validation error of this k (from the cross validation): {round(best_k_error, 2)}")
    print(f"Test error of {best_k}-degree: {round(test_error, 2)}")

def question_7(n_evaluations, train_X, train_y, test_X, test_y):
    reg_parameters = 2 ** np.linspace(-4, 1, n_evaluations)
    ridge_training_errs = np.zeros(n_evaluations)
    ridge_validation_errs = np.zeros(n_evaluations)
    lasso_training_errs = np.zeros(n_evaluations)
    lasso_validation_errs = np.zeros(n_evaluations)
    for i, param in enumerate(reg_parameters):
        # ridge errors
        ridge_training_errs[i], ridge_validation_errs[i] = cross_validate(
            RidgeRegression(param), train_X,
            train_y, scoring=mean_square_error)
        # lasso errors
        lasso_training_errs[i], lasso_validation_errs[i] = cross_validate(
            Lasso(alpha=param, max_iter=10000), np.asarray(train_X),
            np.asarray(train_y), scoring=mean_square_error)

    best_ridge_param = ridge_validation_errs[np.argmin(ridge_validation_errs)]
    best_lasso_param = lasso_validation_errs[np.argmin(lasso_validation_errs)]

    fig = go.Figure([go.Scatter(x=reg_parameters,
                                y=ridge_training_errs,
                                mode='lines+markers',
                                name='Ridge training errors'),
                     go.Scatter(x=reg_parameters,
                                y=ridge_validation_errs,
                                mode='lines+markers',
                                name='Ridge validation errors'),
                     go.Scatter(x=reg_parameters,
                                y=lasso_training_errs,
                                mode='lines+markers',
                                name='Lasso training errors'),
                     go.Scatter(x=reg_parameters, y=lasso_validation_errs,
                                mode='lines+markers',
                                name='Lasso validation errors')])

    fig.update_layout(
        title=f"(Q7) Training and errors as function of regularization parameters (ridge and lasso)",
        title_font_size=15, width=1200, height=700)
    fig.show()

    return best_ridge_param, best_lasso_param

def question_8(n_evaluations, train_X, train_y, test_X, test_y, best_ridge_param, best_lasso_param):
    trained_ridge = RidgeRegression(best_ridge_param)
    trained_ridge.fit(train_X, train_y)
    ridge_error = trained_ridge.loss(test_X, test_y)

    trained_lasso = Lasso(alpha=best_lasso_param, max_iter=50000)
    trained_lasso.fit(train_X, train_y)
    lasso_error = mean_square_error(trained_lasso.predict(test_X), test_y)

    print(f"Ridge test error of {best_ridge_param}-parameter: {ridge_error}")
    print(f"Ridge test error of {best_lasso_param}-parameter: {lasso_error}")


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    # raise NotImplementedError()
    X, y_wo_noise, train_X, train_y, test_X, test_y = generate_poli_dataset(n_samples, noise)

    question_1(X, y_wo_noise, train_X, train_y, test_X, test_y, n_samples, noise)

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    # raise NotImplementedError()

    best_k, best_k_error = question_2(train_X, train_y, n_samples, noise)

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    # raise NotImplementedError()
    question_3(train_X, train_y, test_X, test_y, best_k, best_k_error, n_samples, noise)


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    # raise NotImplementedError()
    train_X, train_y, test_X, test_y = generate_diabetes_dataset(n_samples)


    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    # raise NotImplementedError()
    best_ridge_param, best_lasso_param = question_7(n_evaluations, train_X, train_y, test_X, test_y)

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    # raise NotImplementedError()

    question_8(n_evaluations, train_X, train_y, test_X, test_y,
               best_ridge_param, best_lasso_param)


if __name__ == '__main__':
    np.random.seed(0)
    print("question 1,2,3:")
    select_polynomial_degree()
    print("\nquestion 4:")
    select_polynomial_degree(noise=0)
    print("\nquestion 5:")
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
