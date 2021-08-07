---
title: Stochastic Volatility Models in Stan
date: 2021-08-07
tags: ["volatility", "stan", "bayesian", "model-comparison", "model-building"]
slug: stoch_vol
summary: Fitting 2 different Stochastic Volatility Models to S&P 500 returns and finding out which is better
---

Today I'll be running through Stochastic Volatility Models! These are related to GARCH models in that they allow for time-varying volatility in the return distribution. In other words, it accounts for heteroscedasticity.

# Data

I'm interested in the weekly returns of the S&P 500 index. My intent is to trade weekly options to go short volatility, so weekly forecasts are what I need.


```python
import numpy as np
import pandas as pd
from cmdstanpy import cmdstan_path, CmdStanModel
import matplotlib.pyplot as plt
import arviz as az
from scipy import stats
import statsmodels.api as sm
from psis import psisloo

from datamodel import SPX, StockData

plt.rcParams["figure.figsize"] = (15,10)
```


```python
spx = SPX()
```


```python
spx_wk_prices = spx.prices.resample("W-FRI").last()
spx_wk_returns = (np.log(spx_wk_prices) - np.log(spx_wk_prices.shift(1))).dropna()
```

Here are what the weekly returns look like for the past 20-ish years. You can see that volatility "clusters" meaning that periods of extreme returns are generally followed by periods of extreme returns.


```python
spx_wk_returns.plot(title="S&P 500 Index Weekly Returns", xlabel="Date", ylabel="Return")
```

{{< figure src="./output_7_1.png" >}}


This can be more easily seen with autocorrelation plots. Let's look at the returns themselves first:


```python
fig = sm.tsa.graphics.plot_acf(spx_wk_returns, title="Returns Autocorrelation")
plt.xlabel("Lags")
plt.ylabel("Correlation")
```

{{< figure src="./output_9_1.png" >}}


There's very little autocorrelation, meaning that returns at each time period are unrelated to the returns that came at past time periods. However, let's look at the square of returns, which is a crude way to estimate their volatility.


```python
fig = sm.tsa.graphics.plot_acf(spx_wk_returns**2, title="Squared Returns Autocorrelation")
plt.xlabel("Lags")
plt.ylabel("Correlation")
```

{{< figure src="./output_11_1.png" >}}


Now there is clearly some significant autocorrelation, meaning volatility is affected by past volatility, thus the clustering effect. When there is a volatility shock, we expect to see periods of lasting higher volatility.

# Model 1

I'm going to be fitting a Stochastic Volatility Model which differs from a standard GARCH model. In a GARCH model, variance is modeled as a deterministic function of past errors and past variances:

$$ \sigma_{t}^2 = \omega + \alpha_{1} \epsilon_{t-1}^2 + \beta_{1} \sigma_{t-1}^2 $$

However, in a Stochastic Volatility Model, variance is modeled as a *stochastic* function of past variance:

$$ \sigma_{t}^2 = \mu + \phi (\sigma_{t-1}^2 - \mu) + \epsilon_{t}$$

$$ \epsilon_{t} \sim \mathcal{N}(0, \omega) $$

This model is what is encapsulated below in Stan model language. To use the symbols below it's like this:

$$ r_{t} \sim \mathcal{N}(\mu_{r}, \exp(\frac{h_{t}}{2})) $$

$$ h_{t} = \mu_{h} + \phi (h_{t-1} - \mu_{h}) + \epsilon_{t}$$

$$ \epsilon_{t} \sim \mathcal{N}(0, \sigma) $$

Note here that \\(h\\) represents the logarithm of variance. This makes its distribution far more symmetrical than in its normal form, making fitting the model easier. The gist of the model is that there exists a normal mean variance level represented by \\(\mu_{h}\\) and when shocks occur, whose magnitude is governed by \\(\sigma\\), variance will tend back towards that mean at a rate dictated by \\(\phi\\).

I generate the posterior predictive distribution in the generated quantities block, this will be useful for analysis of the model. I'm using broad uninformative priors here because I have plenty enough data points that they hardly matter.


```python
model_spec = """
    data {
        int N;                              // Length of data
        vector[N] r;                        // SPX returns
    }
    parameters {
        real mu_h;                          // Volatility mean term
        real mu_r;                          // Returns mean term
        real<lower=-1, upper=1> phi;        // Persistence of volatility
        real<lower=0> sigma;                // Volatility noise
        vector[N] h_std;                    // Log volatility
    }
    transformed parameters {
        vector[N] h = h_std * sigma;        // h ~ normal(0, sigma);
        h[1] /= sqrt(1 - square(phi));      // h[1] ~ normal(0, sigma / sqrt(1 - square(phi)))
        h += mu_h;                          // h ~ normal(mu_h, sigma)
        for (t in 2:N) {
            h[t] += phi * (h[t-1] - mu_h);  // h[2:N] ~ normal(mu_h + phi * (h[t-1] - mu_h), sigma)
        }
    }
    model {
        phi ~ uniform(-1, 1);
        sigma ~ normal(0, 10);
        mu_h ~ normal(0, 10);
        mu_r ~ normal(0, 10);
        
        h_std ~ std_normal();
        r ~ normal(mu_r, exp(h / 2));
    }
    generated quantities {
        real r_tilde[N];
        real log_prob[N];
        r_tilde = normal_rng(mu_r, exp(h / 2));
        for (t in 1:N) {
            log_prob[t] = normal_lpdf(r[t] | mu_r, exp(h[t] / 2));  // Need log probabilities later on
        }
    }
"""
with open("./stan_model/model.stan", "w") as file:
    file.write(model_spec)

model = CmdStanModel(stan_file="./stan_model/model.stan")
```

```python
data = {"N": len(spx_wk_returns), "r": spx_wk_returns.values}
sample = model.sample(data=data,
                      chains=4,
                      parallel_chains=4,
                      output_dir="./stan_model",
                      iter_warmup=1000,
                      iter_sampling=2500)
```

```python
model1_data = az.from_cmdstanpy(posterior=sample,
                                posterior_predictive="r_tilde",
                                observed_data={"r": spx_wk_returns.values},
                                log_likelihood="log_prob")
```

First let's look at how our chain sampled to make sure everything looks okay.


```python
az.plot_trace(model1_data, compact=True, var_names=["mu_h", "mu_r", "phi", "sigma"])
```

{{< figure src="./output_19_1.png" >}}


Okay there is no obvious issues here. The parameter distributions from each chain look mostly the same, and there aren't any obvious signs of autocorrelation in the samples. Next, let's look at the summary statistics from our posterior predictive distribution versus our data. The blue histogram bars represent the posterior predictive, and the black line represents that statistic calculated from the data.


```python
r_tilde = model1_data.posterior_predictive.r_tilde.values.reshape(10000, -1)
vol = model1_data.posterior.h.values.reshape(10000, -1)
mean = np.mean(r_tilde, axis=1)
std = np.std(r_tilde, axis=1)
skew = stats.skew(r_tilde, axis=1)
kurt = stats.kurtosis(r_tilde, axis=1)
```


```python
fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(mean, bins=50)
axs[0, 0].axvline(np.mean(spx_wk_returns), color='black')
axs[0, 0].set_title("Mean")
axs[0, 1].hist(std, bins=50)
axs[0, 1].axvline(np.std(spx_wk_returns), color='black')
axs[0, 1].set_title("Standard Deviation")
axs[1, 0].hist(skew, bins=50)
axs[1, 0].axvline(stats.skew(spx_wk_returns), color='black')
axs[1, 0].set_title("Skew")
axs[1, 1].hist(kurt, bins=50)
axs[1, 1].axvline(stats.kurtosis(spx_wk_returns), color='black')
axs[1, 1].set_title("Kurtosis")
```

{{< figure src="./output_22_1.png" >}}    


There are some issues here, right off the bat. Ideally, the black line should fall in a high probability region of the histogram. This would mean that the data simulated from our model closely matches the qualities of the input data. This looks true mostly only for the standard deviation and kurtosis. It seems like the model is not modeling the mean or skew very well. Next let's look at the distribution of our input data versus the distribution of the posterior predictive.


```python
az.plot_ppc(model1_data, data_pairs={"r": "r_tilde"})
```

{{< figure src="./output_24_1.png" >}}


This looks pretty good! The distributions look mostly the same. Next, I want to look at how well calibrated the model is. The model outputs a distributional estimate at each time point. So ideally, for instance, if we calculate the 95th percentile of that distribution, the input data should have values higher than that only 5% of the time. Likewise that data should have values smaller than the 5% percentile only 5% of the time.


```python
# 95% bounds exceedances
np.sum(spx_wk_returns.values > np.percentile(r_tilde, 95, axis=0)) / len(spx_wk_returns)
```

```text
    0.015459723352318959
```

```python
# 5% bounds exceedances
np.sum(spx_wk_returns.values < np.percentile(r_tilde, 5, axis=0)) / len(spx_wk_returns)
```

```text
    0.03986981285598047
```


So 1.6% of the time the data values are above the 95% bounds, and 3.8% of the time the data values are below the 5% bounds. If anything then, our distribution may be too broad. However, in this case that could be considered a good thing because I'd rather predict a broader distribution of returns than a too restrictive one. It's best to be over-prepared for extreme outcomes than under-prepared. The next plot is the 95% and 5% bounds plotted against the return data. You can see the points where the returns exceed those bounds.


```python
plt.plot(np.percentile(r_tilde, 95, axis=0), color="black")
plt.plot(np.percentile(r_tilde, 5, axis=0), color="black")
plt.plot(spx_wk_returns.values, color="red", alpha=0.5)
```

{{< figure src="./output_29_1.png" >}}


The next test is doing a probability integral transform. When you put a value through a CDF it gets transformed onto the range 0 to 1. Ideally, if I put the data through the CDF implied by the model, those output values should be uniformly distributed. This implies that the predicted distribution accurately predicts the probabilities of events. Unlike the exceedances test, which only looks at the tails, this test looks at the entire distribution.


```python
values = []

for t in range(len(spx_wk_returns)):
    ecdf = sm.distributions.empirical_distribution.ECDF(r_tilde[:, t])
    values.append(ecdf(spx_wk_returns.iloc[t]))
```


```python
fig = sm.graphics.qqplot(np.array(values), dist=stats.uniform, line="45")
```

{{< figure src="./output_32_1.png" >}}    


A QQ plot displays the transformed data against a reference distribution. If the samples match a uniform distribution, they should all fall perfectly on the 45 degree line in the figure. It's clear there is some odd behavior at the right tail and in the center. It seems like our distributional estimate doesn't match the data too well.

# Model 2

Okay, so issues are that the mean and skew seem off, and the distribution estimate doesn't match too well with the data. What should I try? Well, we expect negative skew, because large negative returns happen rarely. So instead of assuming a normal error for returns, let's try a skew normal! So everything is the same but the sampling statement for the returns looks like this now:

$$ r_{t} \sim Skew Normal(\mu_{r}, \exp(\frac{h_{t}}{2}), \alpha) $$

Where \\(\alpha\\) is a new parameter that dictates the level of skew. In Stan, that model looks like this.


```python
model_spec = """
    data {
        int N;                              // Length of data
        vector[N] r;                        // SPX returns
    }
    parameters {
        real mu_h;                          // Volatility mean term
        real mu_r;                          // Returns mean term
        real<lower=-1, upper=1> phi;        // Persistence of volatility
        real<lower=0> sigma;                // Volatility noise
        vector[N] h_std;                    // Log volatility
        real alpha;                         // Skew Normal shape parameter
    }
    transformed parameters {
        vector[N] h = h_std * sigma;        // h ~ normal(0, sigma);
        h[1] /= sqrt(1 - square(phi));      // h[1] ~ normal(0, sigma / sqrt(1 - square(phi)))
        h += mu_h;                          // h ~ normal(mu_h, sigma)
        for (t in 2:N) {
            h[t] += phi * (h[t-1] - mu_h);  // h[2:N] ~ normal(mu_h + phi * (h[t-1] - mu_h), sigma)
        }
    }
    model {
        phi ~ uniform(-1, 1);
        sigma ~ normal(0, 10);
        mu_h ~ normal(0, 10);
        mu_r ~ normal(0, 10);
        alpha ~ normal(0, 10);
        
        h_std ~ std_normal();
        r ~ skew_normal(mu_r, exp(h / 2), alpha);
    }
    generated quantities {
        real r_tilde[N];
        real log_prob[N];
        r_tilde = skew_normal_rng(mu_r, exp(h / 2), alpha);
        for (t in 1:N) {
            log_prob[t] = skew_normal_lpdf(r[t] | mu_r, exp(h[t] / 2), alpha);  // Need log probabilities later on
        }
    }
"""
with open("./stan_model/model.stan", "w") as file:
    file.write(model_spec)

model = CmdStanModel(stan_file="./stan_model/model.stan")
```

```python
data = {"N": len(spx_wk_returns), "r": spx_wk_returns.values}
sample = model.sample(data=data,
                      chains=4,
                      parallel_chains=4,
                      output_dir="./stan_model",
                      iter_warmup=1000,
                      iter_sampling=2500)
```

```python
model2_data = az.from_cmdstanpy(posterior=sample,
                                posterior_predictive="r_tilde",
                                observed_data={"r": spx_wk_returns.values},
                                log_likelihood="log_prob")
```

```python
az.plot_trace(model2_data, compact=True, var_names=["mu_h", "mu_r", "phi", "sigma", "alpha"])
```

{{< figure src="./output_39_1.png" >}}

Again, everything looks good here. Alpha centers around a negative value, which is a good sign, because negative skew was expected.

```python
r_tilde = model2_data.posterior_predictive.r_tilde.values.reshape(10000, -1)
vol = model2_data.posterior.h.values.reshape(10000, -1)
mean = np.mean(r_tilde, axis=1)
std = np.std(r_tilde, axis=1)
skew = stats.skew(r_tilde, axis=1)
kurt = stats.kurtosis(r_tilde, axis=1)
```

```python
fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(mean, bins=50)
axs[0, 0].axvline(np.mean(spx_wk_returns), color='black')
axs[0, 0].set_title("Mean")
axs[0, 1].hist(std, bins=50)
axs[0, 1].axvline(np.std(spx_wk_returns), color='black')
axs[0, 1].set_title("Standard Deviation")
axs[1, 0].hist(skew, bins=50)
axs[1, 0].axvline(stats.skew(spx_wk_returns), color='black')
axs[1, 0].set_title("Skew")
axs[1, 1].hist(kurt, bins=50)
axs[1, 1].axvline(stats.kurtosis(spx_wk_returns), color='black')
axs[1, 1].set_title("Kurtosis")
```

{{< figure src="./output_42_1.png" >}}


Now the mean value lies right in the center of the distribution and the skew value is closer to the middle then it was before. That looks like progress!


```python
az.plot_ppc(model2_data, data_pairs={"r": "r_tilde"})
```

{{< figure src="./output_44_1.png" >}}



```python
# 95% bounds exceedances
np.sum(spx_wk_returns.values > np.percentile(r_tilde, 95, axis=0)) / len(spx_wk_returns)
```

```text
    0.025223759153783564
```



```python
# 5% bounds exceedances
np.sum(spx_wk_returns.values < np.percentile(r_tilde, 5, axis=0)) / len(spx_wk_returns)
```

```text
    0.025223759153783564
```


Our exceedances are again a bit too broad but they are more even than the first model.


```python
plt.plot(np.percentile(r_tilde, 95, axis=0), color="black")
plt.plot(np.percentile(r_tilde, 5, axis=0), color="black")
plt.plot(spx_wk_returns.values, color="red", alpha=0.5)
```

{{< figure src="./output_48_1.png" >}}



```python
values = []

for t in range(len(spx_wk_returns)):
    ecdf = sm.distributions.empirical_distribution.ECDF(r_tilde[:, t])
    values.append(ecdf(spx_wk_returns.iloc[t]))
```


```python
fig = sm.graphics.qqplot(np.array(values), dist=stats.uniform, line="45")
```

{{< figure src="./output_50_1.png" >}}


The QQ plot here looks a little bit more funky than the one in model 1, which is concerning.

# Model Comparison

So there are two models. There must be a better way to find out which is better than looking at visualizations. Turns out there is a really cool method for this called Pareto Smoothed Importance Sampling. [This paper covers it very well](https://arxiv.org/abs/1507.04544). It sounds more complicated than it is. It allows the use of posterior samples and log probabilities to estimate the out-of-sample error of the model. It seeks to approximate the error estimated by leave-one-out (LOO) cross validation without running the model repeatedly. In this case, the model doesn't take long to fit, and I could run fewer samples, but it would still have to be run nearly 1200 times to do true LOO cross validation

I'll take a brief aside to discuss leave-one-out methods on time series. [This](https://arxiv.org/abs/1905.11744) paper does some analysis on K-Fold cross validation on time series considering stationarity. They find little difference in error estimation using walk-forward versus K-Fold cross validation on stationary time series. This makes intuitive sense in that stationary time series display no time dependence, so the order in which you use the data shouldn't matter. It's obvious that stock market returns are not stationary. However, for the sake of this analysis, I'm going to assume that they are *conditionally* stationary given the volatility process. This gives some legitimacy to what I'm about to do. I will warn, however, that it's not a perfect method and it's applicability here can be called into question. With all of that said, let's continue.

The author of the first paper linked very nicely has coded this process in python already, available [here](https://github.com/avehtari/PSIS/blob/master/py/psis.py).


```python
model1_probs = model1_data.log_likelihood.log_prob.values.reshape(10000, -1)
model2_probs = model2_data.log_likelihood.log_prob.values.reshape(10000, -1)

loo1, loos1, ks1 = psisloo(model1_probs)
loo2, loos2, ks2 = psisloo(model2_probs)

diff = round(loo2 - loo1, 2)
diff_se = round(np.sqrt(len(loos1) * np.var(loos2 - loos1)), 2)
diff_interval = [round(diff - 2.6 * diff_se, 2), round(diff + 2.6 * diff_se, 2)]
```


```python
print(f"Model 1 ELPD: {round(loo1, 2)}\nModel 2 ELPD: {round(loo2, 2)}")
```

```text
    Model 1 ELPD: 3042.64
    Model 2 ELPD: 3050.58
```


```python
print(f"Model 2 - Model 1: {diff}\nStandard Error: {diff_se}\nDifference 99% Interval: {diff_interval[0]} | {diff_interval[1]}")
```

```text
    Model 2 - Model 1: 7.94
    Standard Error: 3.96
    Difference 99% Interval: -2.36 | 18.24
```

ELPD stands for expected log predictive density. This is what we expect the out-of-sample log probability to be for the model, so we want it to be higher. Higher values imply that the probability of seeing the data given the model is higher, which means the model more closely matches the nature of the data. So, it looks like model 2 *is* better. Although, given the standard error of the estimate, there is some region of the sampling distribution where model 2 is similar or worse but not by very much. This method also returns a value for the shape parameter fitted to the Pareto distribution. Ideally we want this parameter to be less than 0.5 for every point, but 0.5 to 1 is okay. At these higher levels the variance of the estimator is higher and makes it less reliable. Parameter values greater than 1 are highly undesirable. At these levels, the variance of the estimator is infinite and totally unreliable.


```python
ks1_max = round(np.max(ks1), 2)
ks2_max = round(np.max(ks2), 2)

ks1_gt = round(sum(ks1 > 0.5) / len(ks1) * 100, 2)
ks2_gt = round(sum(ks2 > 0.5) / len(ks2) * 100, 2)

print(f"Max k for Model 1: {ks1_max}\nMax k for Model 2: {ks2_max}")
print(f"Percentage of values greater than 0.5 for Model 1: {ks1_gt}%\nPercentage of values greater than 0.5 for Model 2: {ks2_gt}%")
```

```text
    Max k for Model 1: 0.87
    Max k for Model 2: 0.95
    Percentage of values greater than 0.5 for Model 1: 10.9%
    Percentage of values greater than 0.5 for Model 2: 11.64%
```

It looks like the estimates for model 2 are slightly less reliable than model 1, which is worth considering because of how close the difference above was. The max value of k is also quite high, indicating that there are som significant data points making estimation more difficult. All-in-all I would consider model 2 to be better, but the differences are slight.

# Model Volatility versus Realized Volatility

The model basically finds the value of volatility that fits the return data we give it. It's a type of hierarchical model where volatility is a latent quantity. We cannot directly observe the volatility of a return series in the real world, we can only imply it. In the literature, there is a great deal about how to estimate that latent volatility. I've covered a few of those methods in a [previous post](https://eadains.github.io/OptionallyBayesHugo/posts/vol_estimators/). Let's compare what our model thinks volatility is to a realized volatility estimator. I'm going to be taking the volatility from the second model.

Note I'm taking the proper transformations to ensure both series are in standard deviation form. My volatility data is shorter than my weekly returns data, so I have to truncate some of it.


```python
real_vol = np.sqrt(spx.vol.resample("W-FRI").sum())
model_vol = pd.Series(np.mean(arviz_data.posterior.h.values.reshape(10000, -1), axis=0), index=spx_wk_returns.index)
model_vol = np.sqrt(np.exp(model_vol))

common_index = real_vol.index.intersection(model_vol.index)

real_vol = real_vol.loc[common_index]
model_vol = model_vol.loc[common_index]
```


```python
real_vol.plot()
model_vol.plot(color="r")
```

{{< figure src="./output_63_1.png" >}}


With the model volatility in red and the realized measure in blue. The model pretty well captures the realized volatility! It's a smoother estimate, which makes sense considering the linear model for it we are using. Cool!

# Conclusion

The model isn't perfect, but then again no model is! The first one fails to capture the negative skew, and while the second one does better, the QQ plot looks less pleasing. This may mean that in doing better capturing skew, it fails to as effectively capture the middle of the distribution.

There are a lot of interesting extensions you could make to this model. The mean process of volatility could include exogenous regressors like VIX levels, or it could include past values of the returns themselves! Next the volatility of volatility, \\(\sigma\\) in the model, could be made to have a stochastic or deterministic process of its own! Essentially, it could be made to vary with time, just like volatility of the returns.

I'm becoming very interested in Bayesian methods for time series analysis, and there seems to be a lot less literature about that than non-time series models. I think the process for writing, fitting, and interpreting Bayesian models is much more straightforward and clear than frequentist methods. Credible intervals, the fact that parameter uncertainty is automatically accounted for in the posterior predictive distribution, and the methods for estimating out-of-sample error make life much easier.
