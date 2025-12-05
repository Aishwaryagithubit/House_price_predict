
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# 1. Load dataset
data = fetch_california_housing()


# Convert to pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df["MedHouseValue"] = data.target

print(df.head())

# 2. Train-test split
X = df[data.feature_names]
y = df["MedHouseValue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate model
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2) # the more nearer to 1, the best fit the model is.

# 6. Predict a sample input (example)
sample = X_test.iloc[0:1] #  is selecting the first row of the X_test
predicted_value = model.predict(sample)

print("\nSample Input:")
print(sample)
print("\nPredicted Median House Value:", predicted_value[0])
