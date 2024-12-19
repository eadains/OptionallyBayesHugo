---
title: A Most Diversified ETF Portfolio
date: 2021-01-07
tags: ["portfolio-optimization", "ETFs", "diversification-ratio", "performance-measurement", "drawdown"]
summary: Constructing a portfolio from a selection of ETFs to maximize the diversification ratio

resources:
    - alt: test
      source: images/output_24_1.png
---


Alright, with this post I'm going to start a series on portfolio optimization techniques! This is one of my favorite topics in finance. This post is going to construct a portfolio based on the diversification ratio, which is outlined in the papers linked below. The basic idea is to maximize the Diversification ratio, which is defined as the weighted average volatilities of assets in the portfolio divided by the total portfolio volatility. This makes intuitive sense, by increasing diversification we lower portfolio volatility compared to the average volatility of the assets that make it up.

Choueifaty, Y., &amp; Coignard, Y. (2008). Toward Maximum Diversification. The Journal of Portfolio Management, 40-51. doi:[https://doi.org/10.3905/JPM.2008.35.1.40](https://doi.org/10.3905/JPM.2008.35.1.40)

Choueifaty, Y., Reynier, J., &amp; Froidure, T. (2013). Properties of the Most Diversified Portfolio. Journal of Investment Strategies, 49-70. doi:[http://doi.org/10.2139/ssrn.1895459](http://doi.org/10.2139/ssrn.1895459)

# The assets

---


I'm going to mainly focus on Vanguard ETFs as they have the lowest fees. For anything they don't offer, I'm using iShares. I'm also limiting myself to funds with inception dates >10 years ago for stability.

### Here's the list:

| Symbol | Description                                 |
|--------|---------------------------------------------|
| VGSH   | Short-term Treasury                         |
| VGIT   | Mid-term Treasury                           |
| VGLT   | Long-term Treasury                          |
| TIP    | TIPS Treasury Bonds                         |
| VMBS   | Agency MBS                                  |
| SUB    | Municipal Bonds                             |
| VCSH   | Short-term Investment Grade Corporate Bonds |
| VCIT   | Mid-term Investment Grade Corporate Bonds   |
| VCLT   | Long-term Investment Grade Corporate Bonds  |
| HYG    | High-yield Corporate Bonds                  |
| EMB    | Emerging Markets Bonds                      |
| IGOV   | International Treasuries                    |
| VV     | Large Cap US Stocks                         |
| VO     | Mid-Cap US Stocks                           |
| VB     | Small-Cap US Stocks                         |
| VWO    | Emerging Markets Stocks                     |
| VEA    | Non-US Developed Markets Stocks             |
| IYR    | US Real Estate                              |
| IFGL   | Non-US Real Estate                          |

# Data

---

All of this is the code to fetch historical data from QuantConnect and calculate returns.


```python
import numpy as np
import pandas as pd
```


```python
symbols = ['VGSH',
           'VGIT',
           'VGLT',
           'TIP',
           'VMBS',
           'SUB',
           'VCSH',
           'VCIT',
           'VCLT',
           'HYG',
           'EMB',
           'IGOV',
           'VV',
           'VO',
           'VB',
           'VWO',
           'VEA',
           'IYR',
           'IFGL']

qb = QuantBook()
symbols_data = {symbol: qb.AddEquity(symbol) for symbol in symbols}
```


```python
from datetime import datetime
# This is QuantConnect API code to get price history
history = qb.History(qb.Securities.Keys, datetime(2009, 1, 1), datetime(2020, 12, 31), Resolution.Daily)
history = history['close'].unstack(level=0).dropna()
```

I'm using arithmetic returns here so I can easily weight the returns across assets when computing portfolio returns.


```python
returns = (history / history.shift(1)) - 1
returns = returns.dropna()
```


```python
# Let's define some helper functions to get cumulative return series and the total return
def get_cum_returns(returns):
    return (returns + 1).cumprod() - 1

def get_total_return(returns):
    return np.product(returns + 1) - 1
```

# The Optimization

---

This function calculates the diversification ratio for a portfolio given asset weights and their covariance matrix. This is from equation (1) (Choueifaty &amp; Coignard, 2008).


```python
def diverse_ratio(weights, covariance):
    # Standard deviation vector
    stds = np.sqrt(np.diagonal(covariance))
    # Asset-weighted standard deviation
    num = np.dot(weights, stds)
    # Portfolio standard deviation
    denom = np.sqrt(weights @ covariance @ weights)
    return num / denom
```

Now, to confirm that scipy minimize works as we expect for this problem, I'm going to test a bunch of randomized starting weights to confirm that the final weights end up the same. I increase the level of precision using the 'ftol' option because returns are fairly small decimal quantities and I want to ensure the optimization converges completely.


```python
from scipy.optimize import minimize

cov = np.cov(returns.values.T)
# Long-only constraint
bounds = [(0, 1) for x in range(len(cov))]
# Portfolio weights must sum to 1
constraints = (
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
)

results = []
for x in range(100):
    # Set initial weights randomly
    initial = np.random.random(len(cov))
    # Use negative of objective function to maximize
    result = minimize(lambda x, y: -1 * diverse_ratio(x, y), initial, method='SLSQP', args=(cov),
                      bounds=bounds, constraints=constraints, options={'ftol': 1e-10})
    results.append(result.x)
```


```python
# Stack all optimized weight vectors, and round to 4 digits after the decimal
results_array = np.round(np.stack(results), 4)
# Let's look at the standard deviation of the asset weights accross the different optimizations
stds = np.std(results_array, axis=0)
# Looks like they're all zero or nearly zero!
print(stds)
```

```text
    [0.00000000e+00 3.12250226e-17 0.00000000e+00 1.24900090e-16
     0.00000000e+00 0.00000000e+00 0.00000000e+00 4.16333634e-17
     0.00000000e+00 0.00000000e+00 0.00000000e+00 2.08166817e-17
     1.66533454e-16 1.38777878e-16 1.66533454e-16 0.00000000e+00
     0.00000000e+00 1.24900090e-16 2.42861287e-17]
```

Looks like the optimization converges to the same values every time regardless of starting point! This means it's finding the true minimum (or maximum in this case). Let's look the at the weights for each symbol.


```python
# I'll just grab the last optimization result this way
weights_series = pd.Series(data=np.round(result.x, 4), index=returns.columns)
print(weights_series)
```

```text
EMB     0.0000
HYG     0.0215
IFGL    0.0000
IGOV    0.0586
IYR     0.0000
SUB     0.2759
TIP     0.0000
VB      0.0336
VCIT    0.0000
VCLT    0.0000
VCSH    0.0000
VEA     0.0265
VGIT    0.1150
VGLT    0.1868
VGSH    0.1825
VMBS    0.0000
VO      0.0000
VV      0.0817
VWO     0.0179
```


```python
# Let's drop everything with a zero weight
final_weights = weights_series[weights_series > 0]
# Sort by weight for viewing ease
print(final_weights.sort_values(ascending=False))
```

```text
SUB     0.2759
VGLT    0.1868
VGSH    0.1825
VGIT    0.1150
VV      0.0817
IGOV    0.0586
VB      0.0336
VEA     0.0265
HYG     0.0215
VWO     0.0179
```


```python
# Confirm everything sums to 1. Looks good!
final_weights.sum()
```

```text
0.9999999999999999
```


# Backtest Results

---


After doing some offscreen magic to implement this in QuantConnect, we get a dataframe tracking portfolio weights over each month. The algorithm starts at the beginning of 2011 and runs to the end of 2020. On the first trading day of each month it computes the portfolio using the code above using the past year of returns data for each ticker. Note that this is slightly different than the above that uses the entire return history in the optimization.
Let's take a closer look at a particular ETFs portfolio allocation over time. I'm going to use VGSH for this because it's the least risky, most cash-like instrument under consideration.

You can see that the weight changes quite drastically over time, from near zero to nearly 90% in the later parts of 2020. This reflects the nature of how we are calculating asset variances and correlations using only the last 252 days of data. When volatilities or correlations change it causes changes in the allocation.


```python
weights_frame['VGSH'].plot(figsize=(15,10))
```

{{< figure src="./output_24_1.png" align=center >}}
    

In this case, it's caused by a large change in correlation for certain assets in early 2020. Shown in the graph below is the correlation between VGSH and the other ETFs over time. Note the large downward jump on the right side. This shows the weakness of using a rolling data approach like in the backtest. You get big market jumps that dramatically shift your allocation and then when they eventually fall out the backward-looking window, you get big jumps again. I want to come back to this topic some time in the future.


```python
rolling_corr = returns.rolling(252).corr()
rolling_corr['VGSH'].unstack().plot(figsize=(15,10))
```

{{< figure src="./output_26_1.png" align=center >}}
    

```python
# let's reindex the monthly weights frame to daily with forward fill to match returns arra
weights_frame = weights_frame.reindex(returns.index, method='ffill')
# Now we can calculate portfolio returns by weight the returns and summing
port_returns = (weights_frame.values * returns).sum(axis=1, skipna=False).dropna()
# We can also calculate cumulative returns this way because we're working with logarithmic returns
cum_port_returns = get_cum_returns(port_returns)
```

Alright, plotted below are the cumulative returns for the strategy! Note this is without transaction costs factored in.


```python
cum_port_returns.plot(figsize=(15, 10))
```

{{< figure src="./output_29_1.png" align=center >}}


Now let's assemble some backtest statistics. We're going to be using mlfinlab for this task.


```python
from mlfinlab import backtest_statistics

total_return = get_total_return(port_returns)
cagr = (total_return + 1)**(1 / 9) - 1
sharpe = backtest_statistics.sharpe_ratio(port_returns)
drawdown, _ = backtest_statistics.drawdown_and_time_under_water(cum_port_returns + 1)
mar_ratio = cagr / drawdown.max()
```


```python
pd.Series({'Total Return': f'{round(total_return * 100, 2)}%','CAGR': f'{round(cagr * 100, 2)}%', 
           'Sharpe Ratio': round(sharpe, 2), 'Maximum Drawdown': f'{round(drawdown.max() * 100, 2)}%',
           'MAR Ratio': round(mar_ratio, 2)})
```

```text
Total Return        29.77%
CAGR                 2.94%
Sharpe Ratio          1.15
Maximum Drawdown     7.07%
MAR Ratio             0.42
```


Let's compare that to just US large cap stocks over the same period.


```python
vv_returns = returns['VV']['2011':]
vv_cum_returns = get_cum_returns(vv_returns)

total_return = get_total_return(vv_returns)
cagr = (total_return + 1)**(1 / 9) - 1
sharpe = backtest_statistics.sharpe_ratio(vv_returns)
drawdown, _ = backtest_statistics.drawdown_and_time_under_water(vv_cum_returns + 1)
mar_ratio = cagr / drawdown.max()
```


```python
pd.Series({'Total Return': f'{round(total_return * 100, 2)}%','CAGR': f'{round(cagr * 100, 2)}%', 
           'Sharpe Ratio': round(sharpe, 2), 'Maximum Drawdown': f'{round(drawdown.max() * 100, 2)}%',
           'MAR Ratio': round(mar_ratio, 2)})
```



```text
Total Return        267.74%
CAGR                 15.57%
Sharpe Ratio           0.84
Maximum Drawdown     34.28%
MAR Ratio              0.45
```


Looks like the maximum diversification portfolio achieves a higher sharpe ratio! Although it comes at the cost of signficantly lower total returns. More interesting is the MAR ratio, defined as the CAGR over the maximum drawdown. This is a useful ratio because it gauges how much extra return you are getting for taking on heavier drawdown risk. It looks like large cap US stocks win out on this metric.

It gives a different perspective than the Sharpe ratio. The Sharpe ratio uses only standard deviation as a metric for risk. This can be very unrealistic because radically different equity curves can actually have the same Sharpe ratio and total return. That can be interestingly illustrated by reordering returns.


```python
# Okay, let's sort VV returns from least to greatest. Note that these are the same returns, just reordered.
sorted_returns = pd.Series(sorted(vv_returns.values), index=vv_returns.index)
cum_sorted_returns = get_cum_returns(sorted_returns)
# Here you can see the cumulative return graphs. The sorted one looks very unusual, but in fact, the total return ends
# up exactly the same!
cum_sorted_returns.plot(figsize=(15, 10))
vv_cum_returns.plot()
```

{{< figure src="./output_37_1.png" align=center >}}
    

```python
total_return = get_total_return(sorted_returns)
cagr = (total_return + 1)**(1 / 9) - 1
sharpe = backtest_statistics.sharpe_ratio(sorted_returns)
drawdown, _ = backtest_statistics.drawdown_and_time_under_water(cum_sorted_returns + 1)
mar_ratio = cagr / drawdown.max()
```


```python
pd.Series({'Total Return': f'{round(total_return * 100, 2)}%','CAGR': f'{round(cagr * 100, 2)}%', 
           'Sharpe Ratio': round(sharpe, 2), 'Maximum Drawdown': f'{round(drawdown.max() * 100, 2)}%',
           'MAR Ratio': round(mar_ratio, 2)})
```

```text
Total Return        267.74%
CAGR                 15.57%
Sharpe Ratio           0.84
Maximum Drawdown     99.97%
MAR Ratio              0.16
```


As you can see the total return, CAGR, and Sharpe ratio are all the same as the original return series! But the maximum drawdown is *significantly* higher. Obviously this is a worst case scenario, but it shows how drawdowns can drastically affect your portfolio performance over time. Volatility by itself doesn't reflect all kinds of risk because it ignores path dependency. This again is a topic worth covering in more detail at a later date.

# Conclusions

---

Even considering the backtest and attributes of this simple strategy shows deep complexity. In future, I want to compare this optimization strategy to others like the traditional mean-variance approach, hierarchical risk parity, minimum variance, and others. Along with that is discussing extensions like using models to provide forecasts for asset volatility and correlation.

So with all those things to think about, see you next time!
