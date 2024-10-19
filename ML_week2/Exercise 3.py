import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
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
max_depths = [1, 5, 10, 15]  # Depths to be tested
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

# 4. Use Random Forest Regressor
random_forest = RandomForestRegressor(n_estimators=50, random_state=42)

# Train the Random Forest model
random_forest.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = random_forest.predict(X_test)

# Evaluate the Random Forest model using RMSE
rmse_rf = root_mean_squared_error(y_test, y_pred_rf)

print(f"\nRandom Forest Regressor RMSE: {rmse_rf}")

# 5. Plot RMSE comparison between Bagging, Random Forest, and Polynomial Regression
plt.figure(figsize=(10, 6))

plt.plot(max_depths, bagging_rmse, marker='o', label='Bagging Regressor (50 Trees)', color='blue')
plt.axhline(rmse_rf, color='green', linestyle='--', label='Random Forest Regressor (50 Trees)', linewidth=2)
plt.axhline(rmse_poly, color='red', linestyle='--', label='Polynomial Regression (degree=3)', linewidth=2)

plt.xlabel('Max Depth of Decision Trees')
plt.ylabel('Test RMSE')
plt.title('Comparison of Bagging, Random Forest, and Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()

# 6. Scatter plot for Random Forest predictions
def plot_scatter(y_true, y_pred, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, c='green', alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_scatter(y_test, y_pred_rf, 'Random Forest Regressor Predictions')
