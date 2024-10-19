import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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

# 2. Hyperparameter tuning for Decision Tree
# Vary max_depth and splitter, analyze their impact
max_depths = [1, 5, 10, 15]
splitters = ['best', 'random']

# Store the errors for analysis
dt_errors = {}

for splitter in splitters:
    test_errors_depth = []
    for depth in max_depths:
        tree = DecisionTreeRegressor(max_depth=depth, splitter=splitter, random_state=42)
        tree.fit(X_train, y_train)
        y_pred_test = tree.predict(X_test)
        rmse_test = root_mean_squared_error(y_test, y_pred_test) 
        test_errors_depth.append(rmse_test)
        print(f"Decision Tree RMSE (splitter='{splitter}', max_depth={depth}): {rmse_test}")
    
    dt_errors[splitter] = test_errors_depth

# 3. Polynomial regression for comparison
# Create polynomial features and fit a model
poly_features = PolynomialFeatures(degree=3)  # Degree of polynomial can be adjusted
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

# Predict and evaluate polynomial regression
y_pred_poly = poly_model.predict(X_poly_test)
rmse_poly = root_mean_squared_error(y_test, y_pred_poly)

print(f"\nPolynomial Regression RMSE: {rmse_poly}")

# 4. Plot the variation in RMSE based on splitter and max_depth
plt.figure(figsize=(10, 6))
for splitter, errors in dt_errors.items():
    plt.plot(max_depths, errors, label=f'Decision Tree (splitter={splitter})', marker='o')

plt.axhline(rmse_poly, color='red', linestyle='--', label='Polynomial Regression (degree=3)', linewidth=2)

plt.xlabel('Max Depth')
plt.ylabel('Test RMSE')
plt.title('Effect of Splitter on Decision Tree Test RMSE')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot for predicted vs actual values
def plot_scatter(y_true, y_pred, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, c='blue', alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title(title)
    plt.grid(True)
    plt.show()

# Scatter plot for Decision Tree predictions (for best max_depth and splitter)
best_depth = 5  # Choose based on results from the previous analysis
best_splitter = 'best'  # Choose based on results

# Train a Decision Tree model using the chosen max_depth and splitter
best_tree = DecisionTreeRegressor(max_depth=best_depth, splitter=best_splitter, random_state=42)
best_tree.fit(X_train, y_train)
y_pred_tree = best_tree.predict(X_test)

# Scatter plot for Decision Tree predictions
plot_scatter(y_test, y_pred_tree, f'Decision Tree Predictions (max_depth={best_depth}, splitter={best_splitter})')

# Scatter plot for Polynomial Regression predictions
# plot_scatter(y_test, y_pred_poly, 'Polynomial Regression Predictions (degree=3)')

# 5. Brief analysis
# print("\n--- Analysis ---")
# print("1. Effect of max_depth:")
# print("  - Increasing max_depth improves the decision tree's performance initially, but after a point, the model may overfit, causing worse performance.")
# print("2. Effect of splitter:")
# print("  - 'best' splitter tends to perform slightly better as it makes more optimal splits, while 'random' splits can introduce more variance.")
# print("3. Polynomial regression:")
# print("  - Polynomial regression with degree 3 seems to provide a competitive result compared to decision trees with higher max_depth.")
