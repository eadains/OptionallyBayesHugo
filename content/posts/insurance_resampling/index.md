---
title: "Bayesian Method for Insurance Policy Resampling"
date: 2026-01-31
tags: ["python", "bayesian", "insurance", "pymc", "arviz"]
summary: Using pymc to build a model to provide posterior samples of policy-level claim and severity estimates
---

One of the problems with insurance data is that you only get one go at observing losses for a given policy in a given year. You can debate this epistemically, but in my mind the "true" claim frequency or claim severity for a policy in a given year is a latent, unobserved quantity. The only thing you actually observe as an insurer is the policy characteristics, total claim count, and total claim amount. Inferring the true underlying frequency or severity is then a statistical exercise. A useful question to ask is, given the claims we saw, what is a reasonable range of claims we *could* have seen? This is the question I'll provide at least an introduction to here.

# The Inference Model


```python
import arviz as az
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pymc as pm
from scipy import stats
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
```

I'll be using a freely available French auto liability [dataset](https://www.kaggle.com/datasets/floser/french-motor-claims-datasets-fremtpl2freq) for this. There's some preprocessing necessary to get the data together:


```python
df_freq = pl.read_csv("./data/insurance/freMTPL2freq.csv")

df_sev = pl.read_csv("./data/insurance/freMTPL2sev.csv", infer_schema_length=None).group_by("IDpol").sum()

df = df_freq.join(df_sev, on="IDpol", how="left", coalesce=True).with_columns(pl.col("ClaimAmount").fill_null(0))
df_sample = df.sample(5000, seed=42)
```

Now, to infer latent quantities we have to assume a probabilistic model for what we do actually observe. I think this model is about as simple as you can get. The general idea is to first model claim count and claim severity separately. For claim counts we assume they follow a Poisson distribution with a mean equal to the policy exposure times the claim frequency (claims per exposure). We assume claim severity (loss dollars per claim) follows an Exponential distribution. You could easily assume other functional forms here, like log-normal, but for a strictly positive variable with a specified mean, the Exponential distribution is the [maximum entropy distribution](https://en.wikipedia.org/wiki/Maximum_entropy_probability_distribution).

The critical piece here is that we estimate the frequency and severity parameters *individually* for each policy, where we have a hyperprior for both to reduce the total effective number of parameters. This is the notation for the model:
$$
\begin{align*}
C_i &\sim \text{Poisson}(\mu=\lambda_i \text{E}_i) \newline
S_i &\sim \text{Exponential}\left(\lambda=\frac{1}{\mu_i C_i} \right) \newline
\lambda_i &\sim \text{Exponential}(\lambda=\phi) \newline
\mu_i &\sim \text{Exponential}({\lambda=\omega}) \newline
\phi &\sim \text{Exponential}(\lambda=1) \newline
\omega &\sim \text{Exponential}({\lambda=1})
\end{align*}
$$

Where $i$ indexes each policy, $C$ and $S$ represent the *observed* claim count and total loss amount, respectively, and $\text{E}$ represents the policy exposure. By multiplying $\lambda$ and $\text{E}$, we assume that the total number of claims is the policy exposure times the claim frequency, defined as the number of claims per unit of exposure, so we can interpret $\lambda$ as the policy's estimated claim frequency. We set the mean of the Exponential distribution for $S$ to be the observed claim amount $C$ times $\mu$, so we can similarly interpret $\mu$ as the policy's estimated claim severity, the amount of loss dollars incurred per claim.

We can express this in pymc as follows. Note that we only apply the likelihood for the loss amount when it is above 0, because a total loss amount of 0 doesn't tell us anything about the *severity* for the policy, as that happens when we have 0 claims. So, a 0 claim count is informative for our estimate of the claim frequency, but gives us no information about that policy's claim severity, since we didn't actually observe any losses. This becomes a key aspect of how this model works: for claims where we haven't observed any losses we can still take a guess at its claim severity because of the hyperprior.


```python
N = len(df_sample)
E = df_sample["Exposure"].to_numpy()
C = df_sample["ClaimNb"].to_numpy()
S = df_sample["ClaimAmount"].to_numpy()

with pm.Model() as model:
    phi = pm.Exponential("phi", lam=1.0)
    omega = pm.Exponential("omega", lam=1.0)

    lam = pm.Exponential("lambda", lam=phi, shape=N)
    mu = pm.Exponential("mu", lam=omega, shape=N)

    c_obs = pm.Poisson("C", mu=E * lam, observed=C)

    mask = S > 0
    severity_rate = 1.0 / (mu[mask] * C[mask])
    s_obs = pm.Exponential("S", lam=severity_rate, observed=S[mask])

    trace = pm.sample(draws=1000, tune=1000, chains=4)
    trace.extend(pm.sample_posterior_predictive(trace))
    trace.extend(pm.compute_log_likelihood(trace))
```

We can then look at some model diagnostics to see if our MCMC converged well. First, we can look at the chain samples and posterior distribution for our hyperprior variables:


```python
az.plot_trace(trace, var_names=["phi", "omega"])
```

{{< figure src="./insurance_resampling_8_1.png" align=center >}}
    


$\omega$ is not sampling as well as $\phi$ but it looks okay.

Next we can then visualize the energy transition distribution to see if it looks okay:


```python
az.plot_energy(trace)
```

{{< figure src="./insurance_resampling_10_1.png" align=center >}}
    


And the two distributions match, meaning we expect it to have adequately explored the posterior space.

Now, let's look at some diagnostics related to the posterior predictive distributions of $C$ and $S$. First, we expect, a priori, higher exposure policies to contain more information. This is because of a natural assumption about insurance policies that they can be "decomposed" in a certain sense. Here, our exposure is car-years, meaning that if a policy has an exposure of 2, that could be 2 cars for 1 year, or 1 car for 2 years, or some combination. So, one can think of a policy with exposure 2 as being equivalent, from a risk standpoint, as 2 policies both with exposure 1. Holding policy characteristics constant, and glossing over details of deductibles and whatnot, there's no difference between a single policy covering 2 vehicles and 2 separate policies each covering one of them, the expected losses are the same. You can think through the same logic for home insurance and the like.

So, if we think of exposure in this way, it means that policies can be thought of as sums of random variables where each random variable represents a risk with unit exposure. Focusing on claim counts, imagine each of these unit exposure random variables follows a Poisson distribution with mean $\mu$. Then, the total expected claim count for a policy with exposure $N$ is simply $\mu * N$, and, more importantly, the variance of the mean is $\frac{\mu}{\sqrt{N}}$ (from the fact that the variance and mean of a Poisson distribution are equal). Therefore, as $N$, total exposure, increases, we expect the variance of our estimate of the expected total claim count to go down. With some mathematical rearrangement you can see that this same idea applies to an estimate of the claim frequency rather than just total number of claims. Regardless, we can just see if this is true from our posterior samples:


```python
stats.pearsonr(df_sample["Exposure"], az.extract(trace, var_names=["lambda"]).var(axis=1))
```
    PearsonRResult(statistic=np.float64(-0.2568976199074987), pvalue=np.float64(3.488038665730237e-76))



Here we are measuring the degree of correlation between the exposure values and the variance of the samples for the claim frequency for each policy. The negative correlation we observe is exactly as we'd expect: larger exposure policies have a lower variance in their estimated claim frequency.

Next, we can compare the total number of claims in the dataset to our posterior distribution of the same. We expect the observed number of claims to fall in some high probability region of our posterior, which tells us that what we observe matches up with what the model is telling us:


```python
plt.hist(az.extract(trace.posterior_predictive, var_names=["C"]).sum(axis=0), bins=25, color="grey")
plt.axvline(df_sample["ClaimNb"].sum(), color="red")

plt.xlabel("Total Number of Claims")
plt.ylabel("Density")
plt.title("Histogram of Sampled Claim Counts")
plt.legend(["Observed Total", "Sampled Data"])
```

{{< figure src="./insurance_resampling_14_1.png" align=center >}}
    


We can see that things look quite reasonable.

We can do the same analysis for the total loss:


```python
plt.hist(az.extract(trace.posterior_predictive, var_names=["S"]).sum(axis=0), bins=25, color="grey")
plt.axvline(df_sample["ClaimAmount"].sum(), color="red")
plt.xlabel("Total Loss Amount")
plt.ylabel("Density")
plt.title("Histogram of Sampled Loss Amounts")
plt.legend(["Observed Total", "Sampled Data"])
```

{{< figure src="./insurance_resampling_16_1.png" align=center >}}
    


Our observed total loss amount lies a little to the right of the mode of our posterior distribution, but still lies in a reasonable area.

Next, let's compare the posterior claim frequency of a policy where we actually have observed claims to one where we observe zero claims:


```python
# Comparing the posterior distributions of policies with high/low observed frequency
lam_sample = az.extract(trace, var_names="lambda")
high_claim = lam_sample[df_sample["ClaimNb"].to_numpy().argmax()]
low_claim = lam_sample[df_sample["ClaimNb"].to_numpy().argmin()]

plt.hist(high_claim, bins=25, alpha=0.5, color="black")
plt.hist(low_claim, bins=25, alpha=0.5, color="red")
plt.axvline(high_claim.mean(), color="black")
plt.axvline(low_claim.mean(), color="red")

plt.xlabel("Sampled Frequency")
plt.ylabel("Density")
plt.legend(["High Obs Claims", "Low Obs Claims"])
```
    
{{< figure src="./insurance_resampling_18_1.png" align=center >}}
    


You can see that, as we'd expect, when we have actually observed claims for a policy, our estimate of its claim frequency goes up.

From these basic diagnostics, we can see that our model reflects the stylized facts of insurance data that we expect to see. Note that this is a very simple model, you can increase the complexity from here. For example, this model assumes that our estimate of the true latent frequency and severity depends *only* upon our observed values for each, and no other policy characteristics. You can easily see how other specific information about the claim would affect your guess of its true frequency or severity.

# Predictive Model and Lift Charts

I want to demonstrate one potentially useful application of a model like this, which is putting error bounds on lift charts. Typically, in insurance modeling applications, you create a lift chart that compares your modeled predictions of total claims or losses to the actually observed quantities. However, as we said at the beginning, the observed quantities only represent a single observation per policy. By using the posterior we computed above, we can now compute an expected *range* of observed claims and losses, so we can add some error bars to our lift chart, which lets us see probabilistically how our predictions stack up.

Model predictions and actual observations will never perfectly match up on a lift chart, so a natural question to ask is: how far away can my predicted quantity be from the observed before it's unreasonable? And that question is what the lift chart I develop below is able to give some clue to.

First, we develop a very basic lightgbm model to predict claim counts given the policy-specific characteristics in the dataset:


```python
transformer = make_column_transformer(
    (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
        ["Area", "VehBrand", "VehGas", "Region"],
    ),
    ("passthrough", ["Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]),
    remainder="drop",
)

X = transformer.fit_transform(df)
y = df["ClaimNb"].to_numpy()

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.20)

claims_model = lgb.LGBMRegressor(
    learning_rate=0.1,
    n_estimators=10000,
    objective="poisson",
    subsample=0.50,
    subsample_freq=1,
    colsample_bytree=0.5,
    max_depth=6,
    num_leaves=2**6 - 1,
    extra_trees=True,
)
claims_model.fit(
    train_X,
    train_y,
    eval_set=(val_X, val_y),
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
    categorical_feature=[0, 1, 2, 3],
)
```


We can look at some SHAP scatter charts to get a sense of how our input variables affect our predictions:


```python
import shap

explainer = shap.TreeExplainer(claims_model.booster_, feature_names=transformer.get_feature_names_out())
shap_values = explainer(val_X[: int(len(val_X) * 0.10)])

shap.plots.scatter(
    shap_values[:, ["passthrough__Exposure", "passthrough__DrivAge", "passthrough__BonusMalus"]], shap_values
)
```


    
{{< figure src="./insurance_resampling_23_0.png" align=center >}}
    


Then, we can assemble our values for the lift chart. First we compute our quantile bin cutoffs, and then for each bin we compute the total predicted claim count plus some estimated quantiles of observed claim counts using the posterior distribution from the inference model:

(Note that the searchsorted trick here is quite nice. The list of quantiles is sorted by definition, and searchsorted returns the index of the quantiles array for each value of the predictions array that value would have to be placed to maintain its order. So, for instance if quantiles is [1, 2], then any prediction <= 1 would be assigned index 0, anything >1 but <= 2 would be assigned index 1, and anything greater than 2 would be assigned index 2. So, to find all values in the predictions array that are in the first quantile bin, i.e. <1, we just have to find which elements of the array returned by searchsorted are equal to 0, as so on for the other bins.)


```python
predictions = claims_model.predict(transformer.transform(df_sample))

qs = np.linspace(1 / 5, 1, 4, endpoint=False)
quantiles = np.quantile(predictions, qs)
sampled_freqs = az.extract(trace, group="posterior", var_names="lambda")
exposures = df_sample["Exposure"].to_numpy()
actual_claims = df_sample["ClaimNb"].to_numpy()

results = {}
for idx in range(len(quantiles) + 1):
    # This searchsorted trick is quite clever:
    subset_idx = np.searchsorted(quantiles, predictions) == idx
    claims_subset = actual_claims[subset_idx]
    freqs_subset = sampled_freqs[subset_idx]
    exposures_subset = exposures[subset_idx]

    results[idx] = {
        "predicted": claims_subset.sum(),
        # Note that these calculations are the same as multiplying all of the sampled freqs and exposures together and then taking the quantile
        "q25": (np.quantile(freqs_subset, 0.25, axis=1) * exposures_subset).sum(),
        "median": (np.quantile(freqs_subset, 0.50, axis=1) * exposures_subset).sum(),
        "q75": (np.quantile(freqs_subset, 0.75, axis=1) * exposures_subset).sum(),
    }
```


```python
fig, ax = plt.subplots(figsize=(10, 6))

bins = sorted(list(results.keys()))
x_positions = np.arange(len(bins))

medians = [results[bin_idx]["median"] for bin_idx in bins]
q25s = [results[bin_idx]["q25"] for bin_idx in bins]
q75s = [results[bin_idx]["q75"] for bin_idx in bins]
predictions = [results[bin_idx]["predicted"] for bin_idx in bins]

ranges = ax.vlines(
    x_positions, q25s, q75s, color="cornflowerblue", alpha=0.7, linewidth=3, label="25th-75th Percentile Range"
)
median_dots = ax.scatter(x_positions, medians, color="navy", s=80, zorder=3, label="Median")
pred_dots = ax.scatter(x_positions, predictions, color="crimson", marker="X", s=100, zorder=3, label="Predicted")

ax.set_xticks(x_positions)
ax.set_xticklabels([f"Bin {bin_idx}" for bin_idx in bins])

ax.set_xlabel("Bin", fontsize=12)
ax.set_ylabel("Total Claims", fontsize=12)
ax.set_title("Predicted Values vs. Quantile Ranges by Bin", fontsize=14)
ax.legend(handles=[ranges, median_dots, pred_dots], loc="upper left")

ax.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
plt.tight_layout()
```


    
{{< figure src="./insurance_resampling_26_0.png" align=center >}}
    


Here we can see that for each bin, we marked the model predicted total claim count, as well as the 25th, median, and 75th percentile total claim counts computed from our inference model. We can see that for bins 0 through 3, our predictions sit nicely towards the middle of the expected claims distribution. However, for bin 4, our predicted values lies past the quantile boundaries, meaning that we over-predict claim counts for policies in that bin.

This kind of visualization gives you more information about how reasonable your predicted values are compared to the entire posterior distribution of observed claims. Of course, you can do the same kind of thing for total losses, severity, etc. See, for instance, that in bin 1 our predicted value does get away from the median moreso than some of the other bins, but the quantile boundaries tell you that it's not really that off.

# Conclusion

Hopefully this is illustrative, but I think a model like this is very useful for placing your expectations about your historical policy experience in a nice framework. I don't think what we actually observed in the past is the end of the story for insurance, even putting aside development issues, which is another thing you could expand this model to incorporate. The past only gives us a single pass of experience, but we can still use that information to make pretty good guesses about what could have happened.
