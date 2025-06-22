# phase2_model_training.py

# -----------------------------
# Step 1: Import necessary libraries
# -----------------------------
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# Step 2: Load the cleaned dataset
# -----------------------------
print("📥 Loading cleaned dataset...")
df = pd.read_csv("cleaned_expected_ctc.csv")

# Drop ID columns that are not helpful in prediction
df.drop(["IDX", "Applicant_ID"], axis=1, inplace=True)

# -----------------------------
# Step 3: One-Hot Encode categorical variables
# -----------------------------
print("🔄 Performing one-hot encoding on categorical features...")
df_encoded = pd.get_dummies(df, drop_first=True)

# -----------------------------
# Step 4: Split dataset into input (X) and output (y)
# -----------------------------
X = df_encoded.drop("Expected_CTC", axis=1)
y = df_encoded["Expected_CTC"]

# -----------------------------
# Step 5: Train-test split (80% train, 20% test)
# -----------------------------
print("✂️ Splitting into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 6: Train models
# -----------------------------
print("🚀 Training models...")

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# XGBoost
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)

# -----------------------------
# Step 7: Evaluate each model
# -----------------------------
print("\n📊 Model Evaluation Results:")

models = [
    ("Linear Regression", lr_model),
    ("Random Forest", rf_model),
    ("XGBoost", xgb_model)
]

best_model = None
best_score = -np.inf  # For R² comparison

for name, model in models:
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)


    print(f"\n{name}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:,.2f}")

    # Store the best model based on R² score
    if r2 > best_score:
        best_score = r2
        best_model = model

# -----------------------------
# Step 8: Save the best model
# -----------------------------
print("\n💾 Saving the best model (based on R²)...")
with open("salary_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("✅ Best model saved as 'salary_model.pkl'")

# Save feature columns used in training
feature_names = X.columns.tolist()
with open("model_features.pkl", "wb") as f:
    pickle.dump(feature_names, f)
print("✅ Feature names saved as 'model_features.pkl'")

