import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, parse_dates=["Month"], index_col="Month")

# Fit the model
model = ExponentialSmoothing(df["Passengers"], seasonal="add", seasonal_periods=12)
fit = model.fit()

# Add predictions
df["Forecast"] = fit.fittedvalues

# Plot results
df.plot(figsize=(10, 5), title="Airline Passengers Forecast")
plt.ylabel("Passengers")
plt.grid(True)
plt.show()

# Evaluation
mae = mean_absolute_error(df["Passengers"], df["Forecast"])
rmse = np.sqrt(mean_squared_error(df["Passengers"], df["Forecast"]))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
