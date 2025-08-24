# Write a program to demonstrate Regression analysis with residual plots on a given data set.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Step 1: Generate or load dataset (Here we generate synthetic data)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # true relationship: y = 4 + 3x + noise

# Step 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Apply Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict and compute residuals
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# Step 5: Print model performance
print("Intercept:", model.intercept_)
print("Slope:", model.coef_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Step 6: Plot Regression Line
plt.figure(figsize=(10, 5))

# Plot 1: Regression line
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Plot 2: Residual plot
plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted y')
plt.ylabel('Residuals')
plt.tight_layout()
plt.show()
