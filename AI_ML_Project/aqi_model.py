# -------------------------------------------------------------
# Air Quality Index (AQI) Prediction using ML Models
# -------------------------------------------------------------
# This script loads the CPCB dataset (city_day.csv), cleans it,
# selects Delhi data, trains 3 models (Linear, Polynomial, Random
Forest),
# compares their performance, and generates all plots.
# -------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error,
mean_absolute_error
print("Libraries imported")
# -------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------
df = pd.read_csv("city_day.csv")
print("Dataset loaded, shape:", df.shape)
print(df.head())
# -------------------------------------------------------------
# BASIC CLEANING
# -------------------------------------------------------------
# Only keep the columns we actually need
keep_cols = ['City', 'Date', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO',
'O3', 'AQI']
df = df[keep_cols]
# Drop rows where AQI is missing (can't train without target)
df = df.dropna(subset=['AQI'])

# Replace missing pollutant values with column means
df = df.fillna(df.mean(numeric_only=True))
# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
# Extract month and day from the date
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
# Work only with Delhi data (more consistent readings)
df = df[df['City'] == 'Delhi']
print("Cleaned dataset shape:", df.shape)
print(df.head())
# -------------------------------------------------------------
# FEATURE SETUP
# -------------------------------------------------------------
X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Month', 'Day']]
y = df['AQI']
# Split into train/test (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=42
)
print("Train/test split done")
# -------------------------------------------------------------
# LINEAR REGRESSION
# -------------------------------------------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
print("Linear model trained")
# -------------------------------------------------------------
# POLYNOMIAL REGRESSION (degree 3)

# -------------------------------------------------------------
poly_converter = PolynomialFeatures(degree=3)
X_train_p = poly_converter.fit_transform(X_train)
X_test_p = poly_converter.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_train_p, y_train)
y_pred_poly = poly_model.predict(X_test_p)
print("Polynomial model trained")
# -------------------------------------------------------------
# RANDOM FOREST REGRESSION
# -------------------------------------------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest model trained")
# -------------------------------------------------------------
# EVALUATION FUNCTION
# -------------------------------------------------------------
def show_scores(true, pred, model_name):
print("\n", model_name)
print("R2 :", round(r2_score(true, pred), 3))
print("MSE:", round(mean_squared_error(true, pred), 2))
print("MAE:", round(mean_absolute_error(true, pred), 2))
print("-" * 30)
# Compare model performances
show_scores(y_test, y_pred_linear, "Linear Regression")
show_scores(y_test, y_pred_poly, "Polynomial Regression (deg 3)")
show_scores(y_test, y_pred_rf, "Random Forest Regression")
print("Evaluation done")
# -------------------------------------------------------------
# RANDOM FOREST FEATURE IMPORTANCE
# -------------------------------------------------------------

feature_names = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Month',
'Day']
importances = rf_model.feature_importances_
# Sort features by importance
sorted_idx = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 5))
plt.bar(range(len(importances)), importances[sorted_idx],
tick_label=np.array(feature_names)[sorted_idx])
plt.xticks(rotation=45)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()
print("Feature importance saved")
# -------------------------------------------------------------
# RESIDUAL ERROR ANALYSIS
# -------------------------------------------------------------
y_test_np = np.array(y_test)
res_lin = y_test_np - y_pred_linear
res_poly = y_test_np - y_pred_poly
res_rf = y_test_np - y_pred_rf
plt.figure(figsize=(10, 6))
plt.scatter(y_test_np, res_lin, alpha=0.5, label="Linear")
plt.scatter(y_test_np, res_poly, alpha=0.5, label="Polynomial")
plt.scatter(y_test_np, res_rf, alpha=0.5, label="Random Forest")
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Actual AQI")
plt.ylabel("Residual")
plt.title("Residual Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("residuals_comparison.png", dpi=300)
plt.close()

print("Residual plot saved")
# -------------------------------------------------------------
# ACTUAL vs PREDICTED — SCATTER
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear, alpha=0.5, label="Linear")
plt.scatter(y_test, y_pred_poly, alpha=0.5, label="Polynomial")
plt.scatter(y_test, y_pred_rf, alpha=0.5, label="Random Forest")
plt.plot([0, 500], [0, 500], "k--")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.legend()
plt.tight_layout()
plt.savefig("AQI_Comparison_Scatter.png", dpi=300)
plt.close()
print("Scatter plot saved")
# -------------------------------------------------------------
# 100-DAY LINE PLOT
# -------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="Actual", color='black')
plt.plot(y_pred_linear[:100], label="Linear", alpha=0.7)
plt.plot(y_pred_rf[:100], label="Random Forest", alpha=0.7)
plt.title("Model Comparison Over 100 Days (Delhi AQI)")
plt.xlabel("Days")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.savefig("AQI_Comparison_Line.png", dpi=300)
plt.close()
print("All graphs saved successfully")
