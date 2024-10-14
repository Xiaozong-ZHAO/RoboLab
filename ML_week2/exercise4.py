import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate or reuse synthetic data
# Generate synthetic data
x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)

# Flatten the arrays
x1 = x1.flatten()
x2 = x2.flatten()
y = y.flatten()
X = np.vstack((x1, x2)).T

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize AdaBoost Regressor with DecisionTreeRegressor as the base estimator
ada_regressor = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=30),
    n_estimators=100,  # Number of boosting rounds
    random_state=42
)

# Fit the model to the training data
ada_regressor.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = ada_regressor.predict(X_train)

# Compute the mean squared error on the training set
train_mse = mean_squared_error(y_train, y_train_pred)
print(f"Training Mean Squared Error: {train_mse}")

# Compute R² score on the training set
train_r2 = r2_score(y_train, y_train_pred)
print(f"Training R² score: {train_r2}")

# Optional: Visualize True vs Predicted values on the training set
plt.scatter(y_train, y_train_pred, label='Predicted Values', color='blue')

# Add perfect prediction line
perfect_line = np.linspace(min(y_train), max(y_train), 100)
plt.plot(perfect_line, perfect_line, color='red', label='Perfect Prediction')

# Label the plot
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values on Training Set')
plt.legend()
plt.show()