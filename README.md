# Robust Regression
Project work for the Bayesian Data Analysis course.

## Summary
This repo contains:
1. `normal_model.stan`: traditional regression model assuming normality of the residuals.
2. `normal_model_QR_reparametrization.stan`: traditional regression model assuming normality of the residuals, with QR reparametrization of the coefficients to favor convergence.
3. `robust_model.stan`: regression model assuming a t-Student distribution for the residuals.

A working file for these models is provided in `demo.R`

## Problem Definition and Methods

This project shows how to build a Robust Regression model undertaking a Bayesian approach. Models are implemented in <i>Stan</i> and the <i>demo.R</i> file shows how to fit the models and reproduce the results hereby presented. Data description can be found in https://link.springer.com/book/10.1007/978-1-84628-480-9.

A Robust Regression model can be used to build a predictive model when outliers or extreme values are present in the training set. To this extent, we select as training set the 1000 observations highlighited in orange (from a debutanizer column dataset), which correspond to the second half of the available data and exhibit some unusual trends.

<img src="https://user-images.githubusercontent.com/83544651/148597791-d868f19f-29e5-44d3-ba4d-688bd2418e4a.png" width="100%" height="100%">

Two models have been used and compared. The first model is a multiple linear regression model which assumes the normality of the residuals, &epsilon; &sim; N(0, &sigma;): y &sim; N(&alpha; + &beta;x, &sigma;).

The second model is a robust multiple linear regression model. In this model, the normality assumption of the errors is discarded in favor of distributions with fatter tails like the t-distribution, which should be able to better deal with the presence of outliers. In this case the model becomes: y &sim; t(&nu;, &alpha; + &beta;x, &sigma;).

The parameter &nu; gives us the possibility to control the fatness of the tails, hence the weight given to outliers or extreme data present in the data. With regards to the priors, we used weakly informative priors (very weak) for both the intercept, the model coefficients and the scale of the error. Dealing with data that is scaled between 0 and 1, we used:

<ul>
  <li>Intercept: &alpha; &sim; N(0, 10)</li>
  <li>Coefficients: &beta; &sim; N(0, 10)</li>
  <li>Error scale: &sigma; &sim; Inv-&chi;<sup>2</sup>(10)</li>
  <li>Degrees of freedom for t-distribution: &nu; &sim; &chi;<sup>2</sup>(5)
</ul>


## Convergence Diagnostics

For a visual assessment we show the Markov Chains of normal (left) and robust (right) regression parameters (&alpha; and &beta;). It should be noted that all the parameters reported satisfactory R-hat values close to 1.
<img src="https://user-images.githubusercontent.com/83544651/153217137-c537a217-f3cb-4716-891f-9fd1ebad0783.png" width="100%" height="100%">

## Internal Validation

To compare the two models, we first use the LOO scores and the PSIS diagnostic plots where, in order for the models to be correctly specified, all the values of <i>k</i> should be &leq; 0.7. Successively, we compare the two models using y<sub>rep</sub> values, which is an analysis where we are trying to replicate the input data by simulating from the model. We do so by taking as many as many draws from the predictive distribution as the size of the original data. For the
check, y<sub>rep</sub> values are plotted against the true observed values. Plots on the left correspond to the normal regression model while the ones on the right belong to the robust one.

<img src="https://user-images.githubusercontent.com/83544651/153218534-1014b33d-404e-426e-bc11-628e16c30557.png" width="100%" height="100%">

<img src="https://user-images.githubusercontent.com/83544651/153211101-5b5667b3-2a3d-4f58-8b47-d53c3a50fffa.png" width="100%" height="100%">


## External Validation

We can compare the predictions of completely new observations by computing the root mean squared error (RMSE) between the actual values and the ones predicted by the two models. Even in this case the robust regression models outperforms the normal regression model. Indeed, with the first model we obtained an RMSE of 0.210 while with the second model the RMSE is equal to 0.144.

