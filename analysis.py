import numpy as np
import pandas
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import predict
import datetime
from pandas import *
from ggplot import *

def mann_whitney_plus_means(turnstile_weather):

    rain = turnstile_weather['ENTRIESn_hourly'][turnstile_weather.rain == 1]
    norain = turnstile_weather['ENTRIESn_hourly'][turnstile_weather.rain == 0]
    U, p = scipy.stats.mannwhitneyu(rain, norain)
    return rain.mean(), norain.mean(), U, p

def entries_histogram(turnstile_weather):
    plt.figure()

    binBoundaries = np.linspace(0,5000,20)

    turnstile_weather['ENTRIESn_hourly'][turnstile_weather.rain == 0].hist(bins=binBoundaries, label='No rain') 
    turnstile_weather['ENTRIESn_hourly'][turnstile_weather.rain == 1].hist(bins=binBoundaries, label='Rain')

    plt.title('Histogram of ridership distributions during rain and not during rain')
    plt.xlabel('Ridership amount')
    plt.ylabel('Frequency')
    plt.legend()

    return plt

def plot_weekend_data(turnstile_weather):
    f = "%Y-%m-%d"
    turnstile_weather['weekday'] = turnstile_weather.loc[:, 'DATEn'] \
        .apply(lambda x: datetime.strptime(x, f).weekday())
    df = turnstile_weather[['weekday', 'ENTRIESn_hourly']].groupby('weekday').sum().reset_index()
    df.sort('weekday')
    order = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    df['weekday'] = df['weekday'].map(order)
    plt.figure()
    plt.bar(range(len(df['weekday'])), df['ENTRIESn_hourly'], align = 'center')
    plt.xticks(range(len(df['weekday'])), df['weekday'])
    plt.title('Subway ridership by weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Ridership')

    return plt

def plot_residuals(turnstile_weather, predictions):
    
    plt.figure()

    binBoundaries = np.linspace(0,3000,30)

    (turnstile_weather['ENTRIESn_hourly'] - predictions).hist(bins=binBoundaries)

    plt.title('Residual distributions in the regression model')
    plt.xlabel('Residual amount')
    plt.ylabel('Frequency')
    plt.legend()
    return plt
