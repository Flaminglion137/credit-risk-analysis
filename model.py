import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("loan_data_cleaned.csv")

# Encode categorical columns
le = LabelEncoder()
categorical_cols = ["Gender", "Married", "Dependents", "Education",
                    "Self_Employed", "Property_Area", "Loan_Status"]
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=["Loan_ID", "Loan_Status"])
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("=== Model Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Rejected (N)", "Approved (Y)"]))

# Feature importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print("\n=== Feature Importances (ranked) ===")
for _, row in importance_df.iterrows():
    print(f"  {row['Feature']:<22} {row['Importance']:.4f}")

# --- Chart: Feature Importance ---
fig, ax = plt.subplots(figsize=(9, 6))

colors = plt.cm.RdYlGn(
    [i / (len(importance_df) - 1) for i in range(len(importance_df) - 1, -1, -1)]
)

bars = ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1],
               color=colors[::-1], edgecolor="white", height=0.6)

for bar, val in zip(bars, importance_df["Importance"][::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9, color="dimgray")

ax.set_title("Random Forest — Feature Importance for Loan Status Prediction",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Importance Score (Mean Decrease in Impurity)", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)
ax.set_xlim(0, importance_df["Importance"].max() * 1.18)
ax.xaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("\nSaved feature_importance.png")
