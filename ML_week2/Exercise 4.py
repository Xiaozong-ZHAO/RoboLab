import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import root_mean_squared_error

# 1. Generate synthetic data (2D sinusoidal signal)
x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)

# Flatten the arrays
x1 = x1.flatten()
x2 = x2.flatten()
y = y.flatten()
X = np.vstack((x1, x2)).T

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Use AdaBoost Regressor
ada_regressor = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=15),
                                  n_estimators=50,
                                  random_state=42,
                                  loss='linear')

# Train the AdaBoost model
ada_regressor.fit(X_train, y_train)

# Predict on the training set
y_train_pred = ada_regressor.predict(X_train)

# Predict on the test set
y_test_pred = ada_regressor.predict(X_test)

# 3. Compute RMSE on training and test sets
rmse_train = root_mean_squared_error(y_train, y_train_pred)
rmse_test = root_mean_squared_error(y_test, y_test_pred)

print(f"AdaBoost Regressor RMSE (Training Set): {rmse_train}")
print(f"AdaBoost Regressor RMSE (Test Set): {rmse_test}")

# 4. Scatter plot for AdaBoost predictions (Training Set)
def plot_scatter(y_true, y_pred, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, c='blue', alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_scatter(y_train, y_train_pred, 'AdaBoost Regressor Predictions (Training Set)')
