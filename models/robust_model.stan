data {
  int<lower=0> N;                             // number of observations
  int<lower=0> K;                             // number of predictors
  matrix[N, K] x;                             // predictor matrix
  matrix[N, K] x_new;                         // predictor matrix for new inputs
  vector[N] y;                                // outcome vector
}
parameters {
  real alpha;                                 // intercept
  vector[K] beta;                             // coefficients for predictors
  real<lower=0> sigma;                        // error scale
  real<lower=0> nu;                           // t-student df
}
model {
  alpha ~ normal(0, 10);                      // prior for the intercept
  beta ~ normal(0, 10);                       // prior for the coefficients
  sigma ~ inv_chi_square(10);                 // prior for the error scale
  nu ~ chi_square(5);                         // prior for the t-student df
  y ~ student_t(nu, x * beta + alpha, sigma); // likelihood
}
generated quantities{
  real log_lik[N];
  real y_rep[N];
  real y_pred[N];
  for (i in 1:N){
  // posterior predictive replicate
  y_rep[i] = student_t_rng(nu, x[i] * beta + alpha, sigma);
  // log-likelihood
  log_lik[i] = student_t_lpdf(y[i] | nu, x[i] * beta + alpha, sigma);
  // predictions
  y_pred[i] = student_t_rng(nu, x_new[i] * beta + alpha, sigma);
  }
}
