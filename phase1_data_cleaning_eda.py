import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("expected_ctc.csv")  # Make sure the CSV is in the same folder

print("\n🔍 Dataset Info:")
print(df.info())

print("\n📊 First 5 Rows:")
print(df.head())

print("\n🧼 Missing Values:")
print(df.isnull().sum())

df_cleaned = df.dropna()
print(f"\n✅ Cleaned dataset shape: {df_cleaned.shape}")

df_cleaned.to_csv("cleaned_expected_ctc.csv", index=False)
print("✅ Saved cleaned data as 'cleaned_expected_ctc.csv'")

corr = df_cleaned.corr(numeric_only=True)

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")  # Saves the heatmap as an image
print("✅ Correlation heatmap saved as 'correlation_heatmap.png'")
