# -*- coding: utf-8 -*-
"""
# Facebook Prophet
Facebook open-sourced its time-series forecasting tool called Prophet in 2017 which produced accurate forecasts as produced by skilled analysts with a minimum amount of human efforts. The Facebook prophet is available in the form of API in Python and R/

### How Facebook Prophet works?

Facebook Prophet using Additive Regressive models using the following four components:

`y(t) = g(t) + s(t) + h(t) + \epsilon_t`

- **g(t):** A piecewise linear or logistic growth curve trend. Prophet automatically detects changes in trends by selecting change points from the data.
- **s(t):** A yearly seasonal component modeled using the Fourier series and weekly seasonal component using dummy variable
- **h(t):** A user-provided list of important holidays.
- **et**:  Error term used by the prophet.

---
### Install required Python modules
"""

!pip install Cython>=0.22
!pip install cmdstanpy==0.4
!pip install numpy>=1.10.0
!pip install pandas>=0.23.4
!pip install matplotlib>=2.0.0
!pip install LunarCalendar>=0.0.9
!pip install convertdate>=2.1.2
!pip install holidays>=0.9.5
!pip install setuptools-git>=1.2
!pip install python-dateutil>=2.8.0
!pip install pytz>=2017.2
!pip install cycler>=0.10
!pip install kiwisolver>=1.0.1
!pip install pyparsing!=2.0.4
!pip install ephem>=3.7.5.3
!pip install pymeeus
!pip install six
!pip install yfinance

# As of v1.0, the package name on PyPI is “prophet”; prior to v1.0 it was “fbprophet”.
!pip uninstall prophet
!pip install pystan==2.19.1.1
!pip install prophet
##!pip install fbprophet

"""---
### Import required Python modules
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import prophet as fbp
import yfinance as yahooFinance
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Use fivethirtyeight plot style
plt.style.use('fivethirtyeight')

# Test proper installation of Facebook Prophet: https://facebook.github.io/prophet/docs/installation.html
print(f"Facebook Prophet version: { fbp.__version__ }")

"""---
### Load dataset
"""

# Here We are getting Facebook financial information
# We need to pass FB as argument for that
facebook_data = yahooFinance.Ticker("META")

# whole python dictionary is printed here
print(f"Facebook 1y data:")
print( facebook_data.history(period="max") )

# Store the values
filtered_data = facebook_data.history( period="10y" )

df = filtered_data

# Index
##print(f"Index: {df.index}")

# Show the df
print(f"Dataframe:")
df.head()

# add two columnsin dataframe having values as Date and Adj Close
print(f"dataframe:\n{ df }")

# Show `df` keys
print(f"dataframe keys: { df.keys() }")

# add two columnsin dataframe having values as Date and Adj Close
df[['y']] = df[['Close']]

df[ 'ds' ] = df.index

# Convert the 'ds' column to a datetime object without timezone
df[ 'ds' ] = pd.to_datetime(df['ds']).dt.tz_localize(None)

# Subset two columns from data frame
df = df[['ds', 'y']]

df.head()

# Select only `Close` price
#df = df[[ 'Close' ]]

df.head()

"""---
### Dataset splitting: Training and Validation sets
"""

# split data frame  into two parts train and test
split_date = "2019-12-31"

print(f"First element: { df.index[0] }")
print(f"Last element: { df.index[-1] }\n")

df_train = df.loc[ df['ds'] <= split_date].copy()
df_test = df.loc[  df['ds'] > split_date].copy()

print("_" * 80 + "\n")
print(f"Train data:\n{ df_train }")
print("_" * 80 + "\n")
print(f"Test data:\n{ df_test }")

"""---
### Model: Train
"""

# Instantiate Prophet
model = fbp.Prophet()

# Train the model
model.fit( df_train )

"""---
### Forecast
"""

forecast = model.predict(df_test)

forecast.tail()

"""---
### Visualize results
"""

# Plot the data
model.plot(forecast)

# Plot seasonality trends
model.plot_components( forecast )

"""---
### Model evaluation
"""

# code
print( "Mean Squared  Error (MSE):",  round( mean_squared_error(  y_true = df_test["y"], y_pred = forecast['yhat'] ), 2 ) )
print( "Mean Absolute Error (MAE):", round( mean_absolute_error( y_true = df_test["y"], y_pred = forecast['yhat'] ), 2 ) )
