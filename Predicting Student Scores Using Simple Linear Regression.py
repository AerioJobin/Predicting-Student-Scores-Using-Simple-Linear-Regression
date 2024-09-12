# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Load the Dataset
# Replace with the path to your dataset if necessary
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data = pd.read_csv(url)

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# 2. Data Exploration
# Plot the distribution of scores based on study hours
plt.figure(figsize=(6, 4))
plt.scatter(data['Hours'], data['Scores'], color='blue')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.show()

# 3. Data Preparation
X = data[['Hours']]  # Feature (independent variable)
y = data['Scores']   # Target (dependent variable)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Model Prediction
y_pred = model.predict(X_test)

# 6. Model Evaluation
# Calculate mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# 7. Visualizing the Regression Line
# Plot the regression line (on training data)
plt.figure(figsize=(6, 4))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_train, model.predict(X_train), color='red', label='Regression Line')
plt.title('Regression Line (Train Data)')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.legend()
plt.show()

# Plot actual vs predicted values (on test data)
plt.figure(figsize=(6, 4))
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line (Test)')
plt.title('Actual vs Predicted (Test Data)')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.legend()
plt.show()
        
        