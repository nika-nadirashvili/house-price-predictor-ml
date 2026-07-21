import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# Load dataset
data = pd.read_csv("house_data.csv")
# Features and target
X = data[["area", "rooms", "age"]]
y = data["price"]
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# Create model
model = LinearRegression()
# Train model
model.fit(X_train, y_train)
# Test model
predictions = model.predict(X_test)
# Evaluate accuracy
error = mean_absolute_error(
    y_test,
    predictions
)
print("Average prediction error:", error)
# Predict new house
new_house = [[120, 4, 10]]
price = model.predict(new_house)
print("Predicted price:", price[0])
