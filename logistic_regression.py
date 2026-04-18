import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv("loan_data_cleaned.csv")

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

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=== Logistic Regression Results ===")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Rejected (N)", "Approved (Y)"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# --- Chart 5: Confusion Matrix Heatmap ---
fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Rejected (N)", "Approved (Y)"],
    yticklabels=["Rejected (N)", "Approved (Y)"],
    linewidths=0.5,
    linecolor="white",
    annot_kws={"size": 14, "weight": "bold"},
    ax=ax
)

ax.set_title(f"Logistic Regression — Confusion Matrix\nAccuracy: {accuracy:.2%}",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
ax.set_ylabel("True Label", fontsize=11, labelpad=10)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10, rotation=0)

plt.tight_layout()
plt.savefig("chart5.png", dpi=150)
plt.close()
print("\nSaved chart5.png")
