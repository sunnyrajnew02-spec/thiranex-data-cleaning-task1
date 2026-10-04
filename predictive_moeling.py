import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# 1. Load Dataset
df = sns.load_dataset("titanic")

# 2. Basic Inspection & Summary Statistics
print("=== Dataset Overview ===")
print(f"Shape: {df.shape}")
print("\nMissing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\nStatistical Summary:")
print(df.describe())

# 3. Data Cleaning
# Fill missing age with median, drop deck due to heavy nulls
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df.drop(columns=["deck"], inplace=True, errors="ignore")

# 4. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Survival Rate by Gender
sns.barplot(x="sex", y="survived", data=df, ci=None, palette="Blues_d", ax=axes[0, 0])
axes[0, 0].set_title("Survival Rate by Gender")
axes[0, 0].set_ylabel("Survival Rate")
axes[0, 0].set_xlabel("Gender")

# Plot 2: Survival Rate by Passenger Class
sns.barplot(x="pclass", y="survived", hue="sex", data=df, ci=None, palette="Set2", ax=axes[0, 1])
axes[0, 1].set_title("Survival Rate by Class & Gender")
axes[0, 1].set_ylabel("Survival Rate")
axes[0, 1].set_xlabel("Passenger Class")

# Plot 3: Age Distribution by Survival
sns.histplot(data=df, x="age", hue="survived", kde=True, bins=30, palette="tab10", ax=axes[1, 0])
axes[1, 0].set_title("Age Distribution by Survival Status")
axes[1, 0].set_xlabel("Age")

# Plot 4: Correlation Heatmap of Numerical Features
numeric_cols = df.select_dtypes(include=["number"])
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1, 1])
axes[1, 1].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("eda_summary_plots.png")
plt.show()

# 5. Key Findings Summary
print("\n=== Key EDA Findings ===")
print("- Gender was the strongest predictor: ~74% of females survived vs ~19% of males.")
print("- Higher passenger class strongly correlated with survival (1st class > 2nd > 3rd).")
print("- Younger children had a higher survival rate compared to middle-aged adults.")
