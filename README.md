# Robust Regression

This projects shows how to build a Robust Regression model undertaking a Bayesian approach. A Robust Regression model can be used to build a predictive model when outliers or extreme values are present in the training set. To this extent, we select as training set the 1000 observations highlighited in orange (from a debutanizer column dataset), which correspond to the second half of the available data and exhibit some unusual trends.

<img src="https://user-images.githubusercontent.com/83544651/148597791-d868f19f-29e5-44d3-ba4d-688bd2418e4a.png" width="60%" height="60%">

Two models have been used and compared. The first model is a multiple linear regression model which assumes the normality of the residuals, &epsilon; &sim; N(0, &sigma;): y &sim; N(&alpha; + &beta;x, &sigma;).

The second model is a robust multiple linear regression model. In this model, the normality assumption of the errors is discarded in favor of distributions with fatter tails like the t-distribution, which should be able to better deal with the presence of outliers. In this case the model becomes: y &sim; t(&nu;, &alpha; + &beta;x, &sigma;).

The parameter &nu; gives us the possibility to control the fatness of the tails, hence the weight given to outliers or extreme data present in the data. With regards to the priors, we used weakly informative priors (very weak) for both the intercept, the model coefficients and the scale of the error. Dealing with data that is scaled between 0 and 1, we used:

<ul>
  <li>Intercept: &alpha; &sim; N(0, 10)</li>
  <li>Coefficients: &beta; &sim; N(0, 10)</li>
  <li>Error scale: &sigma; &sim; Inv-&chi;<sup>2</sup>(10)</li>
  <li>Degrees of freedom for t-distribution: &nu; &sim; &chi;<sup>2</sup>(5)
</ul>



<img src="https://user-images.githubusercontent.com/83544651/153211101-5b5667b3-2a3d-4f58-8b47-d53c3a50fffa.png" width="60%" height="60%">



