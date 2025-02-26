# **Robust Regression: Bayesian Approach**

This repository contains the project work for the *Bayesian Data Analysis* course, exploring **Robust Regression** using a Bayesian framework. The goal is to compare traditional and robust regression models when dealing with datasets containing **outliers or extreme values**.

## 📌 **Summary**
This repo includes:
1. `normal_model.stan` – Traditional regression model assuming normality of residuals.
2. `normal_model_QR_reparametrization.stan` – Normal regression model with QR reparametrization to improve convergence.
3. `robust_model.stan` – Regression model assuming a **t-distribution** for residuals, improving robustness to outliers.
4. `demo.R` – A working file to fit models and reproduce results.

## 🔍 **Problem Definition & Methodology**
This project builds a **Bayesian Robust Regression model** using *Stan*. The **demo.R** script demonstrates how to fit models and analyze results. Data details can be found at [Springer](https://link.springer.com/book/10.1007/978-1-84628-480-9).

### **Why Robust Regression?**
Standard regression assumes **normally distributed residuals**. However, in real-world datasets, outliers can skew results. A robust approach models errors with a **t-distribution**, which has fatter tails, reducing outlier influence.

📊 **Dataset:** The training set consists of 1000 observations (orange-highlighted), containing unusual trends from a debutanizer column dataset.

<img src="https://user-images.githubusercontent.com/83544651/148597791-d868f19f-29e5-44d3-ba4d-688bd2418e4a.png" width="100%" height="100%">

### **Model Comparison**
1. **Normal Regression:**
   - Assumes residuals follow \( \epsilon \sim N(0, \sigma) \)
   - Model: \( y \sim N(\alpha + eta x, \sigma) \)

2. **Robust Regression (t-distribution):**
   - Allows heavy-tailed errors, reducing outlier impact.
   - Model: \( y \sim t(
u, \alpha + eta x, \sigma) \)
   - Parameter **ν** controls tail fatness.

### **Bayesian Priors**
- \( \alpha \sim N(0, 10) \) (Intercept)
- \( eta \sim N(0, 10) \) (Coefficients)
- \( \sigma \sim Inv-\chi^2(10) \) (Error scale)
- \( 
u \sim \chi^2(5) \) (Degrees of freedom for t-distribution)

## ✅ **Convergence Diagnostics**
Below are the **Markov Chains** for normal (left) and robust (right) regression parameters (α and β). All parameters show satisfactory **R-hat values ≈1**, indicating good convergence.

<img src="https://user-images.githubusercontent.com/83544651/153217137-c537a217-f3cb-4716-891f-9fd1ebad0783.png" width="100%" height="100%">

## 📊 **Model Validation**
### **Internal Validation**
We use **LOO scores** and **PSIS diagnostic plots**. A well-specified model should have **k ≤ 0.7**.

Additionally, we compare the two models using **y_rep values**, where posterior predictive checks are performed by simulating new data points and comparing them with real values.

**Left:** Normal Regression | **Right:** Robust Regression

<img src="https://user-images.githubusercontent.com/83544651/153218534-1014b33d-404e-426e-bc11-628e16c30557.png" width="100%">

<img src="https://user-images.githubusercontent.com/83544651/153211101-5b5667b3-2a3d-4f58-8b47-d53c3a50fffa.png" width="100%">

### **External Validation**
Predicting new observations using **Root Mean Squared Error (RMSE):**
- **Normal Regression:** RMSE = **0.210**
- **Robust Regression:** RMSE = **0.144** *(Lower is better!)*

The **robust regression model outperforms the normal model**, handling outliers more effectively.
