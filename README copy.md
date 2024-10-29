

---

# Solana Price Prediction using XGBoost

This project uses machine learning, specifically an XGBoost regressor, to predict the price of Solana (SOL) based on historical data and engineered features.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Example](#example)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)

## Project Overview

This repository provides a framework for predicting the price of Solana, utilizing time-series data and engineered features like lagged and rolling statistics. The model is trained with the XGBoost algorithm, known for its performance in regression tasks on structured datasets.

## Installation

Ensure you have Python 3.7+ and the following libraries installed:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

## Usage

### 1. Load Data
Load the Solana price data into a DataFrame and parse the `Start` column as the datetime index:

```python
import pandas as pd

# Load dataset
data = pd.read_csv('solana_2020-04-09_2024-10-28.csv')
data['Start'] = pd.to_datetime(data['Start'])
data.set_index('Start', inplace=True)
```

### 2. Feature Engineering
Create lagged and rolling statistical features for the `Close` price:

```python
target = 'Close'
lags = [1, 7, 14, 30]

# Lagged features
for lag in lags:
    data[f'{target}_lag_{lag}'] = data[target].shift(lag)

# Rolling statistics
data['rolling_mean_7'] = data[target].rolling(window=7).mean()
data['rolling_std_7'] = data[target].rolling(window=7).std()
data['rolling_mean_30'] = data[target].rolling(window=30).mean()
data['rolling_std_30'] = data[target].rolling(window=30).std()

# Drop NaN values
data.dropna(inplace=True)
```

### 3. Prepare Features and Target
Separate the features (X) and target variable (y):

```python
X = data.drop(columns=['Close'])
y = data['Close']
```

### 4. Train the XGBoost Model
Configure and train the XGBoost model with categorical support enabled:

```python
from xgboost import XGBRegressor

# Initialize XGBoost Regressor
xgb_model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    enable_categorical=True  # Enable categorical feature support
)

# Fit the model
xgb_model.fit(X, y)
```

### 5. Model Evaluation
Evaluate the model using Mean Absolute Error (MAE) and Mean Squared Error (MSE):

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Predictions
y_pred = xgb_model.predict(X)

# Evaluation metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
```

## Example

An example run to check the dataset and engineered features:

```python
# Display the first few rows of the dataset
print(data.head())
```

## Results

The model's performance can be evaluated by comparing the predicted values against actual values. Here is an example of a quick visualization:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(y.index, y, label="Actual Price")
plt.plot(y.index, y_pred, label="Predicted Price")
plt.xlabel("Date")
plt.ylabel("SOL Price")
plt.legend()
plt.show()
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Let me know if you'd like any adjustments to this outline or additional details in any section!