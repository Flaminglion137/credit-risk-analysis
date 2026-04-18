import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("loan_data_cleaned.csv")

# --- Chart 1: Loan Approval Rate by Gender ---
gender_approval = (
    df.groupby("Gender")["Loan_Status"]
    .apply(lambda x: (x == "Y").sum() / len(x) * 100)
    .reset_index()
)
gender_approval.columns = ["Gender", "Approval_Rate"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(gender_approval["Gender"], gender_approval["Approval_Rate"],
              color=["#4C72B0", "#DD8452"], width=0.5, edgecolor="white")

for bar, rate in zip(bars, gender_approval["Approval_Rate"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{rate:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_title("Loan Approval Rate by Gender", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Gender", fontsize=12)
ax.set_ylabel("Approval Rate (%)", fontsize=12)
ax.set_ylim(0, 100)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("chart1.png", dpi=150)
plt.close()
print("Saved chart1.png")

# --- Chart 2: Loan Approval Rate by Credit History ---
credit_approval = (
    df.groupby("Credit_History")["Loan_Status"]
    .apply(lambda x: (x == "Y").sum() / len(x) * 100)
    .reset_index()
)
credit_approval.columns = ["Credit_History", "Approval_Rate"]
credit_approval["Label"] = credit_approval["Credit_History"].map(
    {0.0: "No Credit History (0.0)", 1.0: "Has Credit History (1.0)"}
)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(credit_approval["Label"], credit_approval["Approval_Rate"],
              color=["#C44E52", "#55A868"], width=0.5, edgecolor="white")

for bar, rate in zip(bars, credit_approval["Approval_Rate"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{rate:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_title("Loan Approval Rate by Credit History", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Credit History", fontsize=12)
ax.set_ylabel("Approval Rate (%)", fontsize=12)
ax.set_ylim(0, 100)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("chart2.png", dpi=150)
plt.close()
print("Saved chart2.png")

# --- Chart 3: Applicant Income Distribution by Loan Status ---
approved = df[df["Loan_Status"] == "Y"]["ApplicantIncome"]
rejected = df[df["Loan_Status"] == "N"]["ApplicantIncome"]

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(
    [approved, rejected],
    labels=["Approved (Y)", "Rejected (N)"],
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker="o", markersize=4, alpha=0.4, linestyle="none"),
)
bp["boxes"][0].set_facecolor("#55A868")
bp["boxes"][1].set_facecolor("#C44E52")

ax.set_title("Applicant Income Distribution by Loan Status", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Loan Status", fontsize=12)
ax.set_ylabel("Applicant Income", fontsize=12)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

for i, (data, label) in enumerate([(approved, "Approved (Y)"), (rejected, "Rejected (N)")], start=1):
    ax.text(i, ax.get_ylim()[1] * 0.97,
            f"Median: {data.median():,.0f}\nn={len(data)}",
            ha="center", va="top", fontsize=9, color="dimgray")

plt.tight_layout()
plt.savefig("chart3.png", dpi=150)
plt.close()
print("Saved chart3.png")
