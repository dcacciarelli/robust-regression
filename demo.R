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

# Loading debutanizer dataset and splitting into train (with extreme values) and test
dat <- read.csv("/Users/dcac/Desktop/PhD/Data/Soft_Sensors/debutanizer.csv")
dat_new = dat %>% slice(which(row_number() <= 2000))
df_train = dat %>% slice(which(row_number() > 1000 & row_number() <= 2000))
df_test = dat %>% slice(which(row_number() <= 1000))

#########################
#                       #
#     Normal Model      #
#                       #
#########################

# Check that modile compiles
stanc("normal_model.stan")
normal_model <- "normal_model.stan"

# Defining data
debutanizer <- list(N=nrow(df_train), K=7, x=as.matrix(df_train[1:7]), y=df_train$y,
                    x_new=as.matrix(df_test[1:7]))

# Fit the model
fit_normal_model <- stan(file=normal_model, data=debutanizer, warmup=1000, iter=2000, chains=5)
print(fit_normal_model)

# Check the convergence by plotting Markov Chains
posterior_normal_model <- as.array(fit_normal_model)
np_normal_model <- nuts_params(fit_normal_model)
color_scheme_set("mix-blue-pink")
mcmc_trace(posterior_normal_model, pars=c("alpha", "beta[1]", "beta[2]", "beta[3]", "beta[4]", "beta[5]", "beta[6]", "beta[7]"), 
           np=np_normal_model, facet_args=list(ncol=4, strip.position="top"))

# Check parameters distribution
stan_dens(fit_normal_model, pars=c("alpha", "beta[1]", "beta[2]", "beta[3]", "beta[4]", "beta[5]", "beta[6]", "beta[7]"), ncol=4)

# Internal validation: LOO and PSIS score
loo_normal_model <- loo(fit_normal_model, save_psis = TRUE)
print(loo_normal_model)
plot(loo_normal_model)

# Yrep values
mcmc_normal_model = as.data.frame(fit_normal_model) %>% dplyr:::select(starts_with("y_rep"))
ppc_dens_overlay(df_train$y, as.matrix(mcmc_normal_model)) + xlim(-0.5, 1)

# External validation
preds_normal_model = as.data.frame(fit_normal_model) %>% dplyr:::select(starts_with("y_pred"))
y_pred_normal_model = colMeans(preds_normal_model)
# Plot
preds_normal_model <- matrix(c(y_pred_normal_model, df_test$y), ncol=2)
matplot(preds_normal_model, type="l", ylab="Butane Content", xlab="Observations")
# RMSE
sqrt(mean((df_test$y - preds_normal_model)^2))


#########################
#                       #
#     Robust Model      #
#                       #
#########################

# Check that modile compiles
stanc("robust_model.stan")
robust_model <- "robust_model.stan"

# Fit the model
fit_robust_model <- stan(file=robust_model, data=debutanizer, warmup=1000, iter=2000, chains=5)
print(fit_robust_model)

# Check the convergence by plotting Markov Chains
posterior_robust_model <- as.array(fit_robust_model)
np_robust_model <- nuts_params(fit_robust_model)
color_scheme_set("mix-blue-pink")
mcmc_trace(posterior_robust_model, pars=c("alpha", "beta[1]", "beta[2]", "beta[3]", "beta[4]", "beta[5]", "beta[6]", "beta[7]"),
           np=np_robust_model, facet_args=list(ncol=4, strip.position="top"))

# Check parameters distribution
stan_dens(fit_robust_model, pars=c("alpha", "beta[1]", "beta[2]", "beta[3]", "beta[4]", "beta[5]", "beta[6]", "beta[7]"), ncol=4)
# Sigma and nu
stan_dens(fit_robust_model, pars=c("sigma"))
mcmc_trace(posterior_robust_model, pars=c("nu"))
stan_dens(fit_robust_model, pars=c("nu"))

# Internal validation: LOO and PSIS score
loo_robust_model <- loo(fit_robust_model, save_psis = TRUE)
print(loo_robust_model)
plot(loo_robust_model)

# Yrep values
mcmc_robust_model = as.data.frame(fit_robust_model) %>% dplyr:::select(starts_with("y_rep"))
ppc_dens_overlay(df_train$y, as.matrix(mcmc_robust_model)) + xlim(-0.5, 1)

# External validation
preds_robust_model = as.data.frame(fit_robust_model) %>% dplyr:::select(starts_with("y_pred"))
y_pred_robust_model = colMeans(preds_robust_model)
# Plot
preds_robust_model <- matrix(c(y_pred_robust_model, df_test$y), ncol=2)
matplot(preds_robust_model, type="l", ylab="Butane Content", xlab="Observations")
# RMSE
sqrt(mean((df_test$y - preds_robust_model)^2))
