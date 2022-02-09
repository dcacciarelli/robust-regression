data {
  int<lower=0> N;                             // number of observations
  int<lower=0> K;                             // number of predictors
  matrix[N, K] x;                             // predictor matrix
  vector[N] y;                                // outcome vector
  matrix[N, K] x_new;                         // predictor matrix for new inputs
}
parameters {
  real alpha;                                 // intercept
  vector[K] beta;                             // coefficients for predictors
  real<lower=0> sigma;                        // error scale
}
model {
  alpha ~ normal(0, 10);                      // prior for the intercept
  beta ~ normal(0, 10);                       // prior for the coefficients
  sigma ~ inv_chi_square(10);                 // prior for the error scale
  y ~ normal(x * beta + alpha, sigma);        // likelihood
}
generated quantities{
  real log_lik[N];
  real y_rep[N];
  real y_pred[N];
  for (i in 1:N){
  // posterior predictive replicate
  y_rep[i] = normal_rng(x[i] * beta + alpha, sigma);
  // log-likelihood
  log_lik[i] = normal_lpdf(y[i] | x[i] * beta + alpha, sigma);
  // predictions
  y_pred[i] = normal_rng(x_new[i] * beta + alpha, sigma);
  }
}
