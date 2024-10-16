#### Developed by LUBINDHER S
#### Register no: 212222240056
#### Date: 
# Ex.No: 6               HOLT WINTERS METHOD
### AIM:

To analyze NVIDIA stock prices and forecast future prices using Holt-Winters exponential smoothing. The goal is to predict the stock's closing prices for the next 30 business days.
### ALGORITHM:
1. Import necessary libraries like pandas, numpy, matplotlib, and ExponentialSmoothing from statsmodels.
2. Load the dataset and parse the 'Date' column as datetime.
3. Set the 'Date' column as the index of the DataFrame.
4. Convert the 'Close' column to numeric and remove rows with missing values.
5. Extract the 'Close' column for time series analysis.
6. Apply the Holt-Winters exponential smoothing model with additive trend and seasonal components.
7. Fit the model to the cleaned data.
8. Forecast the stock prices for the next 30 business days.
9. Plot both the historical stock data and the forecasted prices.
### PROGRAM:
```
# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset
data = pd.read_csv('NVIDIA_Stock_Price.csv')

# Convert the 'Date' column to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Filter the data to include only records from 2015 onward
data = data[data.index >= '2015-01-01']

# Convert 'Close' column to numeric (removing invalid values)
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

# Drop rows with missing values in 'Close' column
clean_data = data.dropna(subset=['Close'])

# Extract 'Close' column for time series forecasting
close_data_clean = clean_data['Close']

# Perform Holt-Winters exponential smoothing
model = ExponentialSmoothing(close_data_clean, trend="add", seasonal="add", seasonal_periods=12)
fit = model.fit()

# Forecast for the next 20 steps (business days)
n_steps = 30
forecast = fit.forecast(steps=n_steps)

# Plot the original data and the forecast
plt.figure(figsize=(10, 6))
plt.plot(close_data_clean.index, close_data_clean, label='Original Data')
plt.plot(pd.date_range(start=close_data_clean.index[-1], periods=n_steps+1, freq='B')[1:], forecast, label='Forecast')  # 'B' for business days
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Holt-Winters Forecast for Google Stock Prices (After 2015)')
plt.legend()
plt.show()
```

### OUTPUT:
![WhatsApp Image 2024-09-27 at 09 35 40_b57ca5cd](https://github.com/user-attachments/assets/ad0741cb-7515-4e43-994b-a4b42446084b)

### RESULT:
The program was successfully completed based on the Holt Winters Method model
