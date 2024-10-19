import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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

# 2. Test Bagging Regressor with varying max_depth
max_depths = [1, 3, 5, 10]  # Depths to be tested
bagging_rmse = []

for depth in max_depths:
    bagging_regressor = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=depth), n_estimators=50, random_state=42)
    bagging_regressor.fit(X_train, y_train)
    
    # Predict and evaluate bagging regressor
    y_pred_bagging = bagging_regressor.predict(X_test)
    rmse_bagging = root_mean_squared_error(y_test, y_pred_bagging)

    bagging_rmse.append(rmse_bagging)
    print(f"Bagging Regressor (max_depth={depth}) RMSE: {rmse_bagging}")

# 3. Polynomial regression for comparison (fixed degree)
poly_features = PolynomialFeatures(degree=3)  # Degree of polynomial can be adjusted
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

# Predict and evaluate polynomial regression
y_pred_poly = poly_model.predict(X_poly_test)
rmse_poly = root_mean_squared_error(y_test, y_pred_poly) 

print(f"\nPolynomial Regression RMSE: {rmse_poly}")

# 4. Plot RMSE comparison between Bagging and Polynomial Regression
plt.figure(figsize=(10, 6))

plt.plot(max_depths, bagging_rmse, marker='o', label='Bagging Regressor (50 Trees)', color='blue')
plt.axhline(rmse_poly, color='red', linestyle='--', label='Polynomial Regression (degree=3)', linewidth=2)

plt.xlabel('Max Depth of Decision Trees')
plt.ylabel('Test RMSE')
plt.title('Comparison of Bagging Regressor RMSE with Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()

# 5. Scatter plot for best Bagging Regressor
best_depth = max_depths[np.argmin(bagging_rmse)]  # Best performing depth based on RMSE

# Train Bagging Regressor with best max_depth
best_bagging_regressor = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=best_depth), n_estimators=50, random_state=42)
best_bagging_regressor.fit(X_train, y_train)
y_pred_best_bagging = best_bagging_regressor.predict(X_test)

# Scatter plot for Bagging predictions
def plot_scatter(y_true, y_pred, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, c='green', alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_scatter(y_test, y_pred_best_bagging, f'Bagging Regressor Predictions (Best max_depth={best_depth})')

# 5. Brief Analysis:
# - Bagging with decision trees provides a lower RMSE compared to a single decision tree or polynomial regression in some cases.
# - By using an ensemble of trees, bagging improves the model's ability to generalize.
