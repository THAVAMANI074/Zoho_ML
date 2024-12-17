# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import set_config
import joblib

# Load the dataset
file_path = "/content/Rotten_Tomatoes_Movies3_1.xlsx"  # Update the file path if needed
data = pd.read_excel(file_path, sheet_name="Rotten_Tomatoes_Movies(3)")

# Display dataset information
print("Dataset Info:")
print(data.info())

# Select only required columns
columns = ["tomatometer_status", "tomatometer_rating", "audience_rating"]
data = data[columns]

# Drop rows where 'audience_rating' is missing (target column)
data = data.dropna(subset=["audience_rating"])

# Features and Target
X = data[["tomatometer_status", "tomatometer_rating"]]
y = data["audience_rating"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numeric and categorical columns
numeric_features = ["tomatometer_rating"]
categorical_features = ["tomatometer_status"]

# Numeric transformer
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Categorical transformer
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine transformers in a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Create a pipeline with Random Forest Regressor
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Visualize pipeline as diagram
set_config(display="diagram")  # Enable pipeline diagram
print("\nMachine Learning Model Pipeline:")
print(model)

# Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Scatter plot of predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="teal")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
plt.xlabel("Actual Audience Rating")
plt.ylabel("Predicted Audience Rating")
plt.title("Actual vs Predicted Audience Ratings")
plt.show()

# Save the model
model_file = "audience_rating_model.pkl"
joblib.dump(model, model_file)
print(f"\nModel saved as '{model_file}'.")
