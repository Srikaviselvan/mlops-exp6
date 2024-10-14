# Import necessary libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

# Load the California housing dataset
california = fetch_california_housing()
X = california.data  # Features
y = california.target  # Target (House Prices)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = MLPRegressor()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)


# Example: predict the price for the first sample in the test set
sample = X_test[0].reshape(1, -1)
predicted_price = model.predict(sample)
print("Training Complete!")
joblib.dump(model, 'model.joblib')

with open("metricts.txt",'w') as fw:
    fw.write(f"Mean Squared Error of current model is: {mean_squared_error(y_test, y_pred)}")