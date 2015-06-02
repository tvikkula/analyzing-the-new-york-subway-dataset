import numpy as np
import pandas
from ggplot import *
import statsmodels.api as sm
import datetime

def normalize_features(df):
    mu = df.mean()
    sigma = df.std()
    
    if (sigma == 0).any():
        raise Exception("One or more features had the same value for all samples," + \
                         + " and thus could not be normalized. Please do not include " + \
                            + "features with only a single value in your model.")
    df_normalized = (df - df.mean()) / df.std()

    return df_normalized, mu, sigma

def compute_cost(features, values, theta):
    m = len(values)
    sum_of_square_errors = np.square(np.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)

    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    m = len(values)
    cost_history = []

    for i in range(num_iterations):
        h = np.dot(features, theta)
        theta = theta + alpha/m * np.dot((values - h), features)
        cost = compute_cost(features, values, theta)
        cost_history.append(cost)
    return theta, pandas.Series(cost_history)

def predictions(dataframe):
    # Select Features (try different features!)
    features = dataframe[['rain', 'precipi', 'Hour', 'meantempi', 'fog', 'meanwindspdi']]
    
    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    
    # Values
    values = dataframe['ENTRIESn_hourly']
    m = len(values)

    features, mu, sigma = normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values)

    # Set values for alpha, number of iterations.
    alpha = 0.2 # please feel free to change this value
    num_iterations = 70 # please feel free to change this value

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array, 
                                                            values_array, 
                                                            theta_gradient_descent, 
                                                            alpha, 
                                                            num_iterations)
    
    plot = None

    plot = plot_cost_history(alpha, cost_history)
    
    predictions = np.dot(features_array, theta_gradient_descent)
    return predictions, plot

def compute_r_squared(data, predictions):
    # returns the coefficient of determination, R^2, for the model that produced
    # predictions.                                                           
    data = data['ENTRIESn_hourly']
    nom = ((data - predictions)**2).sum()
    denom = ((data - np.mean(data))**2).sum()
    rsquared = 1 - (nom/denom)
    return rsquared

def predictions_normal_eq(weather_turnstile):
    f = "%Y-%m-%d"
    weather_turnstile['weekday'] = weather_turnstile['DATEn'] \
        .apply(lambda x: datetime.datetime.strptime(x, f).weekday())
    X = weather_turnstile[['rain', 'precipi', 'meantempi', 'meanpressurei', 'meanwindspdi']]
    X = weather_turnstile[[]]
    dummy_units = pandas.get_dummies(weather_turnstile['UNIT'], prefix='unit')
    X = X.join(dummy_units)
    dummy_units = pandas.get_dummies(weather_turnstile['Hour'], prefix='Hour')
    X = X.join(dummy_units)
    dummy_units = pandas.get_dummies(weather_turnstile['weekday'], prefix='weekday')
    X = X.join(dummy_units)
    y = weather_turnstile['ENTRIESn_hourly']
    features = sm.add_constant(np.array(X))
    model = sm.OLS(np.array(y), features)
    res = model.fit()
    prediction = res.predict()
    return prediction, res

def plot_cost_history(alpha, cost_history):

   cost_df = pandas.DataFrame({
      'Cost_History': cost_history,
      'Iteration': range(len(cost_history))
   })
   return ggplot(cost_df, aes('Iteration', 'Cost_History')) + \
      geom_point() + ggtitle('Cost History for alpha = %.3f' % alpha )


