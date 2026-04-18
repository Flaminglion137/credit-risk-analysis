# credit-risk-analysis


# Who Gets a Loan? Credit Risk Analysis

## Overview
A data analysis project exploring loan approval patterns 
using EDA and logistic regression. Built as part of my 
data analytics portfolio during my gap year.

## Questions I explored
- Do men and women get approved at different rates?
- Does credit history alone predict approval?
- Does income level matter as much as people assume?

## Tools used
Python, Pandas, Matplotlib, Seaborn, Scikit-learn

## Key findings
1. Credit history is everything
Chart2 showed a dramatic difference in approval rates between applicants with and without credit history. The feature importance chart confirmed it credit history scored 0.24, almost double the next most important feature. If you have good credit history, you're very likely to get approved.

2. Income matters but not alone
Chart3 showed approved applicants tend to have higher incomes but there's significant overlap — some low income applicants still got approved and some high income applicants got rejected. This is because LoanAmount matters alongside income — it's the ratio that counts, not the number alone.

5. The top 5 predictors ranked:

Credit History — 0.24
Applicant Income — 0.21
Loan Amount — 0.19
Coapplicant Income — 0.11
Dependents — 0.05

5. The model performed at 86% accuracy
Of 123 test cases it got 106 right. The main weakness was 16 false positives (FP) people predicted to repay who shouldn't have been approved. In a real bank those are the risky customers.

## Dataset
Source: Kaggle — Loan Prediction Dataset

## Live site
(https://flaminglion137.github.io/credit-risk-analysis/)

## How to run
Open the notebook in VS Code or view the live site above
