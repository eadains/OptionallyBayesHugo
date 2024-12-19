---
title: Monte Carlo Methods for Option Pricing and Greeks
date: 2021-05-01
Tags: ["pricing", "pytorch", "options"]
summary: Using PyTorch to easily compute Option Greeks first using Black-Scholes and then Monte Carlo methods.
---

Alright, in this post I'm going to run through how to price options using Monte Carlo methods and also compute the associated greeks using automatic differentiation in PyTorch.

# Black-Scholes

---


First, let's look at implementing the Black-Scholes model in PyTorch.

The input variables are as follows:

\\(K\\) : Strike price of the option

\\(S(t)\\) : Price of the underlying asset at time \\(t\\)

\\(t\\) : Current time in years.

\\(T\\) : Time of option expiration

\\(\sigma\\) : Standard deviation of the underlying *returns*

\\(r\\) : Annualized risk-free rate

\\(N(x)\\) : Standard Normal cumulative distribution function

The price of a call option is given by:

$$C(S_t, t) = N(d_1) S_t - N(d_2) K e^{-r(T-t)}$$

$$d_1 = \frac{1}{\sigma\sqrt{T-t}}[\ln(\frac{S_t}{K}) + (r + \frac{\sigma^2}{2})(T-t)]$$

$$d_2 = d_1 - \sigma\sqrt{T-t}$$

And by parity the price of a put option is given by:

$$P(S_t, t) = N(-d_2) K e^{-r(T-t)} - N(-d_1) S_t$$

---

Now, let's implement that using PyTorch functions. For simplicity I replace \\(T\\) and \\(t\\) and their difference by a single term \\(T\\) specifying the total time left to expiry in years.


```python
import torch
from torch.distributions import Normal

std_norm_cdf = Normal(0, 1).cdf
std_norm_pdf = lambda x: torch.exp(Normal(0, 1).log_prob(x))

def bs_price(right, K, S, T, sigma, r):
    d_1 = (1 / (sigma * torch.sqrt(T))) * (torch.log(S / K) + (r + (torch.square(sigma) / 2)) * T)
    d_2 = d_1 - sigma * torch.sqrt(T)
    
    if right == "C":
        C = std_norm_cdf(d_1) * S - std_norm_cdf(d_2) * K * torch.exp(-r * T)
        return C
        
    elif right == "P":
        P = std_norm_cdf(-d_2) * K * torch.exp(-r * T) - std_norm_cdf(-d_1) * S
        return P
```

With this function I can calculate the price of a call option with the underyling at 100, strike price at 100, 1 year to expiration, 5% annual volatility, and a risk-free rate of 1% annually.


```python
right = "C"
K = torch.tensor(100.0, requires_grad=True)
S = torch.tensor(100.0, requires_grad=True)
T = torch.tensor(1.0, requires_grad=True)
sigma = torch.tensor(0.05, requires_grad=True)
r = torch.tensor(0.01, requires_grad=True)

price = bs_price(right, K, S, T, sigma, r)
print(price)
```

```text
    tensor(2.5216, grad_fn=<SubBackward0>)
```

Now, the magic of PyTorch is that it tracks all of those computations in a graph and can use its automatic differentiation feature to give us all the greeks. That's why I told it that I needed a gradient on all of the input variables.


```python
# Tell PyTorch to compute gradients
price.backward()
```


```python
print(f"Delta: {S.grad}\nVega: {sigma.grad}\nTheta: {T.grad}\nRho: {r.grad}")
```

```text
    Delta: 0.5890103578567505
    Vega: 38.89707946777344
    Theta: 1.536220908164978
    Rho: 56.379390716552734
```

How do these compare to the greeks computed directly by differentiating the Black-Scholes formula?


```python
d_1 = (1 / (sigma * torch.sqrt(T))) * (torch.log(S / K) + (r + (torch.square(sigma) / 2)) * T)
d_2 = d_1 - sigma * torch.sqrt(T)

delta = std_norm_cdf(d_1)
vega = S * std_norm_pdf(d_1) * torch.sqrt(T)
theta = ((S * std_norm_pdf(d_1) * sigma) / (2 * torch.sqrt(T))) + r * K * torch.exp(-r * T) * std_norm_cdf(d_2)
rho = K * T * torch.exp(-r * T) * std_norm_cdf(d_2)
```


```python
print(f"Delta: {delta}\nVega: {vega}\nTheta: {theta}\nRho: {rho}")
```

```text
    Delta: 0.5890103578567505
    Vega: 38.89707946777344
    Theta: 1.5362210273742676
    Rho: 56.379390716552734
```

Exactly the same to a high level of precision! Amazing. It's easy to see how much simpler the PyTorch autograd approach is. Note that it is possible to calculate second-order derivatives like Gamma, it just requires remaking the computation graph. If anyone knows of a workaround to this let me know.


```python
S = torch.tensor(100.0, requires_grad=True)
price = bs_price(right, K, S, T, sigma, r)

delta = torch.autograd.grad(price, S, create_graph=True)[0]
delta.backward()

print(f"Autograd Gamma: {S.grad}")

# And the direct Black-Scholes calculation
gamma = std_norm_pdf(d_1) / (S * sigma * torch.sqrt(T))
print(f"BS Gamma: {gamma}")
```

```text
    Autograd Gamma: 0.07779412716627121
    BS Gamma: 0.0777941569685936
```

# Monte Carlo Pricing

---


Now that's all fine, but nothing new except some computation tricks. Black-Scholes makes assumptions that can often violate what is observed in the real world. The problem is creating closed form pricing models under other market dynamics is usually impossible. That's where Monte Carlo sampling comes in. It's a trivial task to create future market paths given a model for its dynamics. You can calculate option payoffs from those paths and get a price. But how can you calculate greeks from Monte Carlo samples? Again, PyTorch and autograd can help.

I'll use all of the same parameters as in the example above. Let's simulate the result of a Geometric Brownian Motion process after one year, just like Black-Scholes does.


```python
K = torch.tensor(100.0, requires_grad=True)
S = torch.tensor(100.0, requires_grad=True)
T = torch.tensor(1.0, requires_grad=True)
sigma = torch.tensor(0.05, requires_grad=True)
r = torch.tensor(0.01, requires_grad=True)

Z = torch.randn([1000000])
# Brownian Motion
W_T = torch.sqrt(T) * Z
# GBM
prices = S * torch.exp((r - 0.5 * torch.square(sigma)) * T + sigma * W_T)
```


```python
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 10)

plt.hist(prices.detach().numpy(), bins=25)
plt.xlabel("Prices")
plt.ylabel("Occurences")
plt.title("Distribution of Underlying Price after 1 Year")
```

{{< figure src="./output_18_1.png" align=center >}}
    


Now, let's calculate the option payoffs under each of those future prices, discount them using the risk-free rate, and then take the mean to get the option price. The price calculated with this method is close to the price calculated using Black-Scholes.


```python
payoffs = torch.max(prices - K, torch.zeros(1000000))
value = torch.mean(payoffs) * torch.exp(-r * T)
print(value)
```

```text
    tensor(2.5215, grad_fn=<MulBackward0>)
```

Now, the magic comes in. The only random sampling I used above was a parameter-less standard normal. This fact allows PyTorch to keep track of gradients throughout all of the calculations above. This is called a Pathwise Derivative. This means we can use autograd just like above to get greeks.


```python
value.backward()
```


```python
print(f"Delta: {S.grad}\nVega: {sigma.grad}\nTheta: {T.grad}\nRho: {r.grad}")
```

```text
    Delta: 0.5890941023826599
    Vega: 38.89133834838867
    Theta: 1.536162257194519
    Rho: 56.38788604736328
```

All the same! This means that we can simulate any Monte Carlo process we want, as long as its random component can be reparameterized, and get prices and greeks. Obviously this is a trivial example, but let's look at a more complicated path-dependent option contract like an Asian Option. This type of option has a payoff based on the *average* price of the underlying over it's duration, rather than only the price at expiration like a Vanilla Option. This means we must simulate the price movement each day instead of just at the end.


```python
# All the same parameters for the price process
K = torch.tensor(100.0, requires_grad=True)
S = torch.tensor(100.0, requires_grad=True)
T = torch.tensor(1.0, requires_grad=True)
sigma = torch.tensor(0.05, requires_grad=True)
r = torch.tensor(0.01, requires_grad=True)

dt = torch.tensor(1 / 252)
Z = torch.randn([1000000, int(T * 252)])

# Brownian Motion
W_t = torch.cumsum(torch.sqrt(dt) * Z, 1)
# GBM
prices = S * torch.exp((r - 0.5 * torch.square(sigma)) * T + sigma * W_t)
```


```python
plt.plot(prices[0, :].detach().numpy())
plt.xlabel("Number of Days in Future")
plt.ylabel("Underlying Price")
plt.title("One Possible Price path")
plt.axhline(y=torch.mean(prices[0, :]).detach().numpy(), color="r", linestyle="--")
plt.axhline(y=100, color='g', linestyle="--")
```

{{< figure src="./output_26_1.png" align=center >}}
    


The payoff of an Asian Option given this price path is the difference between the strike price, the green dashed line, and the daily average price over the year, shown by the dashed red line. In this case, the payoff would be zero because the average daily price is below the strike.


```python
# Payoff is now based on mean of underlying price, not terminal value
payoffs = torch.max(torch.mean(prices, axis=1) - K, torch.zeros(1000000))
#payoffs = torch.max(prices[:, -1] - K, torch.zeros(100000))
value = torch.mean(payoffs) * torch.exp(-r * T)
print(value)
```

```text
    tensor(1.6765, grad_fn=<MulBackward0>)
```


```python
value.backward()
```


```python
print(f"Delta: {S.grad}\nVega: {sigma.grad}\nTheta: {T.grad}\nRho: {r.grad}")
```

```text
    Delta: 0.6314291954040527
    Vega: 20.25724220275879
    Theta: 0.5357358455657959
    Rho: 61.46644973754883
```

PyTorch Autograd once again gives us greeks even though we are now pricing a totally different contract. Awesome!

# Conclusion

---


Monte Carlo methods provide a way to price options under a much broader range of market process models. However, computing greeks can be challenging, either having to use finite difference methods or calculating pathwise derivatives symbolically. Using PyTorch can mitigate those issues and use automatic differentiation to provide greeks straight out of the box with no real overhead.
