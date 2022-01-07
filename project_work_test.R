library(aaltobda)
library(loo)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
library(bayesplot)
library(rstanarm)
library(ggplot2)
library(mcmcplots)
library(dplyr)

# QR Reparametrization
write("
data {
  int<lower=0> N;                               // number of data items
  int<lower=0> K;                               // number of predictors
  matrix[N, K] x;                               // predictor matrix
  vector[N] y;                                  // outcome vector
}
transformed data {
  matrix[N, K] Q_ast;
  matrix[K, K] R_ast;
  matrix[K, K] R_ast_inverse;
  // thin and scale the QR decomposition
  Q_ast = qr_thin_Q(x) * sqrt(N - 1);
  R_ast = qr_thin_R(x) / sqrt(N - 1);
  R_ast_inverse = inverse(R_ast);
}
parameters {
  real alpha;                                   // intercept
  vector[K] theta;                              // coefficients on Q_ast
  real<lower=0> sigma;                          // error scale
}
model {
  y ~ normal(Q_ast * theta + alpha, sigma);     // likelihood
}
generated quantities {
  vector[K] beta;
  real log_lik[N];
  real y_rep[N];
  beta = R_ast_inverse * theta; // coefficients on x
  for (i in 1:N){
  // posterior predictive replicate
  y_rep[i] = normal_rng(x[i] * beta + alpha, sigma);
  // log-likelihood
  log_lik[i] = normal_lpdf(y[i] | x[i] * beta + alpha, sigma);
  }
}",

"Multiple_Linear_Regression_QR.stan")

stanc("Multiple_Linear_Regression_QR.stan")

QR <- "Multiple_Linear_Regression_QR.stan"
fit_QR <- stan(file=QR, data=debutanizer, warmup=1000, iter=2000, chains=5)
print(fit_QR)

"
As described in QR-reparameterization section, if you do not have an informative 
prior on the location of the regression coefficients, then you are better off 
reparameterizing your model so that the regression coefficients are a generated 
quantity. In that case, it usually does not matter much what prior is used on on
the reparameterized regression coefficients and almost any weakly informative 
prior that scales with the outcome will do.
"

posteriorQR <- as.array(fit_QR)
npQR <- nuts_params(fit_QR)
color_scheme_set("mix-blue-pink")
mcmc_trace(posteriorQR, pars=c("alpha", "beta[1]", "beta[2]", "beta[3]",
                              "beta[4]", "beta[5]", "beta[6]", "beta[7]"),
           np=npQR)

stan_dens(fit_QR, pars=c("alpha", "beta[1]", "beta[2]", "beta[3]",
                          "beta[4]", "beta[5]", "beta[6]", "beta[7]"))

as.data.frame(t(rhat(fit_QR))) %>% dplyr:::select("alpha", starts_with("beta"))

looQR <- loo(fit_QR, save_psis = TRUE)
print(looQR)
plot(looQR)

mcmcQR = as.data.frame(fit_QR) %>% dplyr:::select(starts_with("y_rep"))
ppc_dens_overlay(dat$y, as.matrix(mcmcQR)) + xlim(-0.5, 1)

dat <- read.csv("/Users/dcac/Desktop/PhD/Data/Soft_Sensors/debutanizer.csv")
matplot(dat[8], type="l", ylab="Butane Content", xlab="Observations")

# extreme values in the training set
dat_new = dat %>% slice(which(row_number() <= 2000))
df_train = dat %>% slice(which(row_number() > 1000 & row_number() <= 2000))
df_test = dat %>% slice(which(row_number() <= 1000))

plot.ts(df_train, main="Train set with extreme values")
plot.ts(df_test, main="Test set")

plot.ts(rbind(df_train, df_test), main="Test set")

# df_test = dat %>% slice(which(row_number() > 1000 & row_number() <= 2000))
# df_train = dat %>% slice(which(row_number() <= 1000))

# take 1000 for train and 1000 for test
# df_train = dat %>% slice(which(row_number() %% 2 == 0 & row_number() <= 2000))
# df_test = dat %>% slice(which(row_number() %% 2 == 1 & row_number() <= 2000))

# Predictors
matplot(df_train[1:7], type="l", ylab="Process Variables", xlab="Observations")
matplot(df_train[8], type="l", ylab="Butane Content", xlab="Observations")
matplot(df_test[8], type="l", ylab="Butane Content", xlab="Observations")


plot.ts(dat_new, main="")

# Output Variable
matplot(df_train[8], type="l", ylab="Butane Content", xlab="Observations")
# ggplot(data=dat, aes(x=seq(1, nrow(dat), 1), y=y)) + 
#   geom_line(color="red", size=1) +
#   geom_point(color="red", size=3) + 
#   labs(x = "Observations", y = "Butane Content")


# Model

# Simple Model
write("
data {
  int<lower=0> N;                             // number of observations
  int<lower=0> K;                             // number of predictors
  matrix[N, K] x;                             // predictor matrix
  vector[N] y;                                // outcome vector
  matrix[N, K] x_new;                         // predictor matrix for new inputs
  //vector[N] y_pred;                                // outcome vector
}
parameters {
  real alpha;                                 // intercept
  vector[K] beta;                             // coefficients for predictors
  real<lower=0> sigma;                        // error scale
}
model {
  alpha ~ normal(0, 1);                      // prior for the intercept
  beta ~ normal(0, 1);                       // prior for the coefficients
  sigma ~ inv_chi_square(1);                 // prior for the error scale
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
}",

"Multiple_Linear_Regression.stan")

#y_rep = we are drawing as many draws from the predictive distribution as the size of the original data.
#kinda replicating the data by simulating from the model

stanc("Multiple_Linear_Regression.stan")

MLR <- "Multiple_Linear_Regression.stan"
debutanizer <- list(N=nrow(df_train), K=7, x=as.matrix(df_train[1:7]), y=df_train$y,
                    x_new=as.matrix(df_test[1:7]))
fit_mlr <- stan(file=MLR, data=debutanizer, warmup=1000, iter=2000, chains=5)
print(fit_mlr)

posterior1 <- as.array(fit_mlr)
np1 <- nuts_params(fit_mlr)
color_scheme_set("mix-blue-pink")
mcmc_trace(posterior1, pars=c("alpha", "beta[1]", "beta[2]", "beta[3]",
                              "beta[4]", "beta[5]", "beta[6]", "beta[7]"),
           np=np1, facet_args=list(ncol=4, strip.position="top"))

stan_dens(fit_mlr, pars=c("alpha", "beta[1]", "beta[2]", "beta[3]",
                                   "beta[4]", "beta[5]", "beta[6]", "beta[7]"),
          ncol=4)



df1 = as.data.frame(t(rhat(fit_mlr))) %>% dplyr:::select("alpha", starts_with("beta"))
print(df1)

loo1 <- loo(fit_mlr, save_psis = TRUE)
print(loo1)
plot(loo1)

mcmc = as.data.frame(fit_mlr) %>% dplyr:::select(starts_with("y_rep"))
ppc_dens_overlay(df_train$y, as.matrix(mcmc)) + xlim(-0.5, 1)

# Predictions
preds = as.data.frame(fit_mlr) %>% dplyr:::select(starts_with("y_pred"))
Y_pred = colMeans(preds)
# Plot
df_preds <- matrix(c(Y_pred, df_test$y), ncol=2)
# RMSE
sqrt(mean((df_test$y - Y_pred)^2))

matplot(df_preds, type="l", ylab="Butane Content", xlab="Observations")


pp_check.foo <- function(object, type = c("multiple", "overlaid"), ...) {
  type <- match.arg(type)
  y <- object[["y"]]
  yrep <- object[["yrep"]]
  stopifnot(nrow(yrep) >= 50)
  samp <- sample(nrow(yrep), size = ifelse(type == "overlaid", 50, 5))
  yrep <- yrep[samp, ]
  
  if (type == "overlaid") {
    ppc_dens_overlay(y, yrep, ...)
  } else {
    ppc_hist(y, yrep, ...)
  }
}
x <- list(y = dat$y, yrep = as.matrix(mcmc))
class(x) <- "foo"
pp_check(x, type = "multiple", binwidth = 0.3)
pp_check(x, type = "overlaid")

# Robust noise model
write("
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
  real<lower=0> nu;                           // df t-student
}
model {
  alpha ~ normal(0, 1);                      // prior for the intercept
  beta ~ normal(0, 1);                       // prior for the coefficients
  sigma ~ inv_chi_square(1);                 // prior for the error scale
  nu ~ chi_square(1);                         // prior for the Student-t df
  y ~ student_t(nu, x * beta + alpha, sigma);     // likelihood
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
}",

"Robust_Regression.stan")

stanc("Robust_Regression.stan")

RR <- "Robust_Regression.stan"
debutanizer_RR <- list(N=nrow(df_train), K=7, x=as.matrix(df_train[1:7]), 
                       y=df_train$y, x_new=as.matrix(df_test[1:7]))#, nu=5)
fit_RR <- stan(file=RR, data=debutanizer_RR, warmup=1000, iter=2000, chains=5)
print(fit_RR)

posterior2 <- as.array(fit_RR)
np2 <- nuts_params(fit_RR)
mcmc_trace(posterior2, pars=c("alpha", "beta[1]", "beta[2]", "beta[3]",
                              "beta[4]", "beta[5]", "beta[6]", "beta[7]"),
           np=np2, facet_args=list(ncol=4, strip.position="top"))

stan_dens(fit_RR, pars=c("alpha", "beta[1]", "beta[2]", "beta[3]",
                          "beta[4]", "beta[5]", "beta[6]", "beta[7]"),
          ncol=4)

stan_dens(fit_RR, pars=c("sigma"))

mcmc_trace(posterior2, pars=c("nu"))
stan_dens(fit_RR, pars=c("nu"))


df = as.data.frame(t(rhat(fit_RR))) %>% dplyr:::select("alpha", starts_with("beta"))
print(df)

loo2 <- loo(fit_RR, save_psis = TRUE)
print(loo2)
plot(loo2)

mcmc2 = as.data.frame(fit_RR) %>% dplyr:::select(starts_with("y_rep"))
ppc_dens_overlay(df_train$y, as.matrix(mcmc2)) + xlim(-0.5, 1)


# Predictions
preds = as.data.frame(fit_RR) %>% dplyr:::select(starts_with("y_pred"))
Y_pred = colMeans(preds)
# Plot
df_preds <- matrix(c(Y_pred, df_test$y), ncol=2)
# RMSE
sqrt(mean((df_test$y - Y_pred)^2))

matplot(df_preds, type="l", ylab="Butane Content", xlab="Observations")

loo_compare(loo1, loo2)


x2 <- list(y = dat$y, yrep = as.matrix(mcmc2))
class(x2) <- "foo"
pp_check(x2, type = "multiple", binwidth = 0.3)
pp_check(x2, type = "overlaid")



# 2nd Model
write("
data {
  int<lower=0> N;                             // number of observations
  int<lower=0> K;                             // number of predictors
  matrix[N, K] x;                             // predictor matrix
  vector[N] y;                                // outcome vector
  real<lower=0> nu;                           // df t-student
}
transformed data {
  matrix[N, K] Q_ast;
  matrix[K, K] R_ast;
  matrix[K, K] R_ast_inverse;
  // thin and scale the QR decomposition
  Q_ast = qr_thin_Q(x) * sqrt(N - 1);
  R_ast = qr_thin_R(x) / sqrt(N - 1);
  R_ast_inverse = inverse(R_ast);
}
parameters {
  real alpha;                                   // intercept
  vector[K] theta;                              // coefficients on Q_ast
  real<lower=0> sigma;                          // error scale
}
model {
  y ~ student_t(nu, Q_ast * theta + alpha, sigma);     // likelihood
}
generated quantities {
  vector[K] beta;
  real log_lik[N];
  real y_rep[N];
  // coefficients on x
  beta = R_ast_inverse * theta; 
  for (i in 1:N){
  // posterior predictive replicate
  y_rep[i] = student_t_rng(nu, Q_ast[i] * theta + alpha, sigma);
  // log-likelihood
  log_lik[i] = student_t_lpdf(y[i]  | nu, Q_ast[i] * theta + alpha, sigma);
  }
}",

"Robust_Regression.stan")

stanc("Robust_Regression.stan")

RR <- "Robust_Regression.stan"
debutanizer_RR <- list(N=nrow(dat), K=7, x=as.matrix(dat[1:7]), y=dat$y, nu=10)
fit_RR <- stan(file=RR, data=debutanizer_RR, warmup=1000, iter=2000, chains=5)

