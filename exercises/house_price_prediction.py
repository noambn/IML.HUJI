from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.metrics.loss_functions import mean_square_error

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

# columns by type:
ZERO_AND_ABOVE = ["bathrooms", "floors", "sqft_above", "sqft_basement",
                  "yr_renovated"]
ONLY_POSITIVE = ["price", "sqft_living", "sqft_lot", "floors", "yr_built",
                 "sqft_living15", "sqft_lot15"]


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # raise NotImplementedError()
    full_data = pd.read_csv(filename).drop_duplicates()
    data = full_data.drop(['id', 'date', 'lat', 'long'],
                          axis=1)
    data = data.dropna()

    for f in ZERO_AND_ABOVE:
        data = data[data[f] >= 0]

    for f in ONLY_POSITIVE:
        data = data[data[f] > 0]

    data['yr_renovated'] = np.where(data['yr_renovated'] == 0.0,
                                    data['yr_built'], data['yr_renovated'])

    data = pd.get_dummies(data, columns=['zipcode'],
                          drop_first=True)

    features, label = data.drop("price", axis=1), data['price']

    return features, label


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # raise NotImplementedError()
    p_corr_all = []
    y_std = np.std(y)
    for f in X:
        f_std = np.std(X[f])
        f_corr = (np.cov(X[f], y)[0][1] / (f_std * y_std))
        p_corr_all.append((f, f_corr))

        fig = go.Figure().add_trace(go.Scatter(x=X[f], y=y, mode='markers',
                                               marker=dict(color="black")))
        fig.update_xaxes(title_text=f"Feature {f}", title_font_size=15)
        fig.update_yaxes(title_text="House price", title_font_size=15)
        fig.update_layout(
            title_text=f"Peareson correlation of feature {f} and price (is {f_corr})",
            title_font_size=30, width=1200, height=700)
        # fig.show()

        file_name = f"/correlation_of_{f}_and_price.pdf"
        fig.write_image(output_path + file_name)

    print(p_corr_all[0])


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    # raise NotImplementedError()
    features, label = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # raise NotImplementedError()
    # feature_evaluation(features, label, "figures")
    # Question 3 - Split samples into training- and testing sets.
    # raise NotImplementedError()
    train_X, train_y, test_X, test_y = split_train_test(features, label, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
    all_mean_loss = []
    all_std_loss = []
    p_range = np.linspace(10, 101, 92)
    for p in range(10, 101):
        loss_of_p = []
        for i in range(10):
            sample_X, sample_y = train_X.sample(frac=p / 100,
                                                random_state=i), train_y.sample(
                frac=p / 100, random_state=i)
            estimator = LinearRegression()
            estimator.fit(sample_X.to_numpy(), sample_y.to_numpy())
            loss_of_p.append(
                estimator.loss(test_X.to_numpy(), test_y.to_numpy()))
        mean_loss, std_loss = np.mean(loss_of_p), np.std(loss_of_p)
        all_mean_loss.append(mean_loss)
        all_std_loss.append(std_loss)

    confidence_interval_plus = np.asarray(all_mean_loss) + 2 * np.asarray(
        all_std_loss)
    confidence_interval_minus = np.asarray(all_mean_loss) - 2 * np.asarray(
        all_std_loss)

    fig = go.Figure((go.Scatter(x=p_range, y=all_mean_loss,
                                mode="markers+lines", name="Mean Prediction",
                                line=dict(dash="dash"),
                                marker=dict(color="green", opacity=.7)),
                     go.Scatter(x=p_range, y=confidence_interval_minus,
                                fill=None, mode="lines",
                                line=dict(color="lightgrey"),
                                showlegend=False),
                     go.Scatter(x=p_range, y=confidence_interval_plus,
                                fill='tonexty', mode="lines",
                                line=dict(color="lightgrey"),
                                showlegend=False)))
    fig.update_layout(
        title_text=f"(Question 4) MSE as function of training data size",
        xaxis_title='Training data percentage', yaxis_title='MSE',
        title_font_size=30, width=1200, height=700)
    fig.show()
