import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # raise NotImplementedError()
    data = pd.read_csv(filename,
                       parse_dates=['Date']).drop_duplicates().dropna()
    data['DayOfYear'] = data['Date'].dt.day_of_year
    for f in ['Month', 'Day', 'Year']:
        data = data[data[f] > 0]
    data = data[data['Month'] < 12]
    data = data[data['Day'] < 32]
    data = data[data['Temp'] > -20]

    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    # raise NotImplementedError()
    df = load_data("../datasets/City_Temperature.csv")
    features, label = df.drop('Temp', axis=1), df['Temp']

    # Question 2 - Exploring data for specific country
    # raise NotImplementedError()
    df_only_israel = df[df.Country == 'Israel']

    df_only_israel['Year'] = df_only_israel['Year'].astype(
        str)  # makes the data discrete

    fig = px.scatter(df_only_israel, x='DayOfYear', y='Temp', color='Year')
    fig.update_layout(
        title=f'(Question 2 - a) Daily temperature as function of day of the year',
        title_font_size=20, width=1200, height=700,
        xaxis_title='Day of the year', yaxis_title='Temperature')
    fig.show()

    df_israel_by_month = df_only_israel.groupby('Month').agg('std')
    fig2 = px.bar(df_israel_by_month, y='Temp')
    fig2.update_layout(
        title=f'(Question 2 - b) Standard deviation of the daily temperatures by month',
        title_font_size=20, width=1200, height=700,
        yaxis_title='Standard deviation of temperature')
    fig2.show()

    # Question 3 - Exploring differences between countries
    # raise NotImplementedError()
    df_by_country_month = df.groupby(['Country', 'Month']).Temp.agg(
        ['mean', 'std']).reset_index()
    fig3 = px.line(df_by_country_month, x='Month', y='mean', error_y='std',
                   color='Country')
    fig3.update_layout(
        title=f'(Question 3) Average monthly temperature color coded by the country (with standard deviation)',
        title_font_size=20, width=1200, height=700,
        yaxis_title='Average')
    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    # raise NotImplementedError()
    train_X, train_y, test_X, test_y = split_train_test(
        df_only_israel['DayOfYear'], df_only_israel['Temp'], 0.75)

    loss_by_k = []
    for k in range(1, 11):
        pf = PolynomialFitting(k)
        pf.fit(train_X.to_numpy(), train_y.to_numpy())
        k_loss = round(pf.loss(test_X.to_numpy(), test_y.to_numpy()), 2)
        print(f'MSE for {k}:', k_loss)
        loss_by_k.append(k_loss)

    x = np.linspace(1, 10, 10)
    fig4 = px.bar(x=x, y=loss_by_k)
    fig4.update_layout(title=f'(Question 4) Loss by polynomial model degree',
                       title_font_size=20, width=1200, height=700,
                       xaxis_title='Polynom degree',
                       yaxis_title='Loss')
    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    # raise NotImplementedError()
    pf = PolynomialFitting(5)
    pf.fit(df_only_israel['DayOfYear'], df_only_israel['Temp'])
    error_by_country = []
    countries = ['Jordan', 'South Africa', 'The Netherlands']
    for country in countries:
        df_country = df[df.Country == country]
        country_loss = round(pf.loss(df_country["DayOfYear"], df_country['Temp']), 2)
        error_by_country.append(country_loss)

    fig5 = px.bar(x=countries, y=error_by_country)
    fig5.update_layout(title=f'(Question 5) Loss by country over Israel data trained model',
                       title_font_size=20, width=1200, height=700,
                       xaxis_title='Country',
                       yaxis_title='Loss')
    fig5.show()
