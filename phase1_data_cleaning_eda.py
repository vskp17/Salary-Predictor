# phase1_data_cleaning_eda.py

# ----------------------------------------
# Step 1: Import necessary libraries
# ----------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------
# Step 2: Load the dataset
# ----------------------------------------
df = pd.read_csv("expected_ctc.csv")  # Make sure the CSV is in the same folder

# ----------------------------------------
# Step 3: Basic dataset info
# ----------------------------------------
print("\n🔍 Dataset Info:")
print(df.info())

print("\n📊 First 5 Rows:")
print(df.head())

# ----------------------------------------
# Step 4: Check for missing values
# ----------------------------------------
print("\n🧼 Missing Values:")
print(df.isnull().sum())

# ----------------------------------------
# Step 5: Drop rows with missing values
# ----------------------------------------
df_cleaned = df.dropna()
print(f"\n✅ Cleaned dataset shape: {df_cleaned.shape}")

# ----------------------------------------
# Step 6: Save cleaned data for next steps
# ----------------------------------------
df_cleaned.to_csv("cleaned_expected_ctc.csv", index=False)
print("✅ Saved cleaned data as 'cleaned_expected_ctc.csv'")

# ----------------------------------------
# Step 7: EDA – Correlation heatmap (numeric columns only)
# ----------------------------------------
corr = df_cleaned.corr(numeric_only=True)

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")  # Saves the heatmap as an image
print("✅ Correlation heatmap saved as 'correlation_heatmap.png'")
