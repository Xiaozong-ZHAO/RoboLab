import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# 1. Load the California Housing dataset
california_data = fetch_california_housing()
X, y = california_data.data, california_data.target

# 2. Preprocess the data (optional: standardize/normalize features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 3. Use AdaBoost Regressor with DecisionTreeRegressor as base estimator
ada_regressor = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=30),  # Adjust depth as needed
                                  n_estimators=100,  # Number of boosting rounds
                                  random_state=42,
                                  loss='linear')

# Train the AdaBoost model
ada_regressor.fit(X_train, y_train)

# Predict on the training set
y_train_pred = ada_regressor.predict(X_train)

# Predict on the test set
y_test_pred = ada_regressor.predict(X_test)

# 4. Compute RMSE on training and test sets
rmse_train = root_mean_squared_error(y_train, y_train_pred)
rmse_test = root_mean_squared_error(y_test, y_test_pred)

print(f"AdaBoost Regressor RMSE (Training Set): {rmse_train:.4f}")
print(f"AdaBoost Regressor RMSE (Test Set): {rmse_test:.4f}")

# 5. Scatter plot for AdaBoost predictions (Test Set)
def plot_scatter(y_true, y_pred, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, c='blue', alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_scatter(y_test, y_test_pred, 'AdaBoost Regressor Predictions (Test Set)')
