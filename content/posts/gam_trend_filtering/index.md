---
title: "Extending Trend Filtering to the GAM Case"
date: 2024-12-18
tags: ["gam", "trend-filtering", "non-parametric-regression", "machine-learning"]
summary: Extending the trend filtering approach to the multivariate additive case
---

In one of my previous posts I implemented a trend filtering model in the univariate case. This is useful on its own but I want to extend it to the multivariate additive case to make it more useful for real-world modeling. Here I'll consider this model form:
$$
y_i = \alpha + f_1(x_{i, 1}) + f_2(x_{i, 2}) + \ldots + f_k(x_{i, k}) + \epsilon
$$
So, we're assuming that the value of $y$ is a linear function of functions of each of our input variables $x$. In this case each of the smoothing functions, $f_j$ will be fit using the trend filtering method. So, this is a traditional GAM where we're changing the form of the smoothing functions. I'll be again using a least squares fit for simplicity, so we're assuming $\epsilon$ is a standard normal random variable, but this probabilistic interpretation won't matter much here because I'll be focusing more on implementation.

# Synthetic Data

I'll be using synthetic data with a known additive functional form to make sure that the model fitting procedure can properly recover the true underlying function. The function is simple:
$$
\begin{align*}
y_i &= 5 + \sin(x_{i, 1}) + \exp(x_{i, 2}) + \epsilon \newline
\epsilon &\sim \text{Normal}(\mu = 0, \sigma^2 = 0.5^2)
\end{align*}
$$

I'm also simulating the $x$ values in a specific way to ensure some behavior in the values to make sure I can handle common practical issues. Firstly, the total number of values I'm simulating is 10,000, but I'll be sampling integers bounded in such a way that the total number of unique possibilities is smaller than the total number of values I'm sampling. By the pigeonhole principle, this guarantees that I will sample multiple of the same value which ensures that my fitting procedure can deal with data where there are repeated $x$ values. For example, I'm sampling $x_{i, 1}$ as integers from -100 to +100 and then dividing by 10 to get rational values between -10 and 10. There are only 200 unique values possible, but again, I'm sampling 10,000 points, and this guarantees repeated values. In other words, the number of unique values will always be less than 10,000.

Secondly, I'm also making sure that the number of unique values differs between each of the $x$'s. For $x_{i, 1}$ there will be at most 200 unique values, and for $x_{i, 2}$ there will be at most 500. Technically, by random chance they could be the same, but this is *exceedingly* unlikely to happen given 10,000 sample points, so I'm ignoring this possibility. This behavior ensures that my fitting procedure can deal with each input feature having different numbers of unique values, necessitating individual treatment which we'll see later.


```python
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import scipy
import cvxpy as cp

rng = np.random.default_rng()
pio.renderers.default = "iframe"
```


```python
n = 10000
X = np.hstack(
    [
        rng.integers(-100, 100, size=(n, 1)) / 10,
        rng.integers(-250, 250, size=(n, 1)) / 250,
    ]
)
true_y = 5 + np.sin(X[:, 0]) + np.exp(X[:, 1])
obs_y = true_y + 0.5 * rng.standard_normal(n)
```

We can then plot our true function and some sampled values:


```python
plot_x = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
plot_y = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)

x_grid, y_grid = np.meshgrid(plot_x, plot_y)

Z = scipy.interpolate.griddata(
    (X[:, 0], X[:, 1]), true_y, (x_grid, y_grid), method="linear"
)
```


```python
fig = go.Figure(
    data=[
        go.Surface(x=plot_x, y=plot_y, z=Z),
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=obs_y,
            opacity=0.15,
            mode="markers",
            marker={"size": 3, "color": "black"},
        ),
    ]
)
fig.update_layout(title="True Function with Sampled Values")
fig.show()
```

{{< iframe figure_4.html 550 >}}

We have our data now, so we can move on to fitted our trend filtering model. What we want to do is fit a separate smoothing function to each variable, so the below code creates a dictionary of dictionary that contains all of the information we need to keep track of for each variable to then construct our model.

Keeping track of each variable separately immediately solves one of the problems outline above, namely that of differing numbers of unique values per input variable. The next challenge is dealing with the presence of duplicate values in our input data. The trend filtering fitting procedure will only work if our input data points are sorted and do not have any duplicate values. To handle this I construct two arrays for each variable: first is a sorted array containing all of the unique values, and the second is a reconstruction array containing indices that can reconstruct the entire original array of observations from the array of unique values.

This way we can then create our $D$ matrix which is used for applying to penalty to the parameters, as well as a fitted parameter vector, $\beta$, the entires of which correspond to each unique observation value. In the model, then, we can reconstruct an array of equal length to our original observation vector by indexing $\beta$ using the reconstruction indices we've precomputed.


```python
def make_D_matrix(n):
    ones = np.ones(n)
    return scipy.sparse.spdiags(np.vstack([-ones, ones]), range(2), m=n - 1, n=n)


params = {}

for i in range(X.shape[1]):
    unique_vals, recon_idx = np.unique(X[:, i], return_inverse=True)
    params[i] = {
        "sort_idx": np.argsort(X[:, i]),
        # These are guaranteed to be sorted
        "unique_vals": unique_vals,
        "recon_idx": recon_idx,
        "D_mat": make_D_matrix(len(unique_vals)),
        "beta_vec": cp.Variable(len(unique_vals), name=f"X_{i}"),
    }
```

So we have now precomputed the things we need to assemble our model. First, we create a variable for the intercept. Next, we can get our model predicted values by taking the intercept plus the values from the fitted $\beta$ vector reassembled by using the reconstruction array for each input variable. Then, we can compute the penalty for each input variable by taking each of their $D$ matrices and matrix multiplying it with the corresponding $\beta$ vector, norming, and summing.

In notation:
$$
\hat{y}_i = \alpha + \sum_{j=1}^k \beta_{i,j}
$$

Where $k$ is the total number of input variables, and $\beta_{i,j}$ is the fitted value corresponding to data point $i$ for variable $j$. The array indexing I do below in the code is needed to fetch the correct $\beta$ vector value for each data point, given that we have duplicated values and our original data is not sorted.

Our total penalty term looks like this:
$$
P = \sum_{j=1}^k \Vert D_j \beta_j \Vert_1
$$

This is just the sum of the $\ell_1$ norm of the difference matrix applied to the parameter vector for each variable. See my last post on the univariate trend filtering case for more details about how this works. Here we are simply applying the univariate trend filtering penalty to each variable individually and combining them.

So, now that we have our model predicted values for each input as well as the penalty term, we can assemble it all together into our objective function, which is a simple least squares objective with a penalty term:
$$
\text{argmin}_{\alpha, \beta} \frac{1}{2} \Vert y - \hat{y} \Vert^2_2 + \lambda P
$$

where $\lambda$ is a free regularization parameter to be selected. From these equations you can start to see the appeal of this method: there is only 1 hyperparameter to be dealt with. Unlike splines, you don't have to worry about selecting knots, because the trend filtering process does this implicitly for us.

There is only one more detail to be dealt with which is identifiability. This issue is already well known in the larger GAM literature, and the solution is simple, although not necessarily complete, as I'll discuss later. The problem is that the space spanned by our smoothing functions includes a constant function, which means that each smoothing function also implicitly includes its own intercept term, along with the one we've explicitly added. This results in a problem where the intercept is not identifiable because the model is equivalent from a loss perspective whether the necessary constant terms get added to the intercept we've specified or whether it gets added to any of the implicit intercepts of any of the smoothing functions. For example, our synthetic data has a true intercept of 5, but our model may end up setting the intercept term to 6.5, and then offset this by moving one of the smoothing functions down by 1.5 everywhere. There are an infinite number of these possibilities, which means our model is not identifiable.

To fix this we add a constraint which is commonly used in the literature, which is to require that the total effect of each smoothing function across the entire input space sums to zero:
$$
\sum_i f(x_{i, j}) = 0 \quad \forall j
$$

In our case, because the effect of our smoothing functions is totally defined by the $\beta$ vectors this simplifies to:
$$
\sum_i \beta_{i, j} = 0 \quad \forall j
$$

This effectively constrains the implicit intercept of each smoothing function to be zero, which solves the identifiability problem, with some caveats.

Putting all of this together, we can finally fit our model:


```python
# For each observed y value get the relevant beta coefficient for that X observation
# by using the reconstruction index based on the unique values vector
alpha = cp.Variable(name="alpha")
y_hat = alpha + cp.sum(
    [params[i]["beta_vec"][params[i]["recon_idx"]] for i in params.keys()]
)
# Compute separate l1 norms for each input variable and sum
penalty = cp.sum(
    [cp.norm(params[i]["D_mat"] @ params[i]["beta_vec"], 1) for i in params.keys()]
)

lam = 5
objective = cp.Minimize(0.5 * cp.sum_squares(obs_y - y_hat) + lam * penalty)
# Sum to zero constraint to fix identifiability problems
constraints = [cp.sum(params[i]["beta_vec"]) == 0 for i in params.keys()]
prob = cp.Problem(objective, constraints)
results = prob.solve(solver="CLARABEL")
```

Then we can compare our fitted function to the true function:


```python
Z_fitted = scipy.interpolate.griddata(
    (X[:, 0], X[:, 1]), y_hat.value, (x_grid, y_grid), method="nearest"
)

fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"is_3d": True}, {"is_3d": True}]],
    subplot_titles=[
        "True Function",
        "Fitted Piecewise Function",
    ],
)

fig.add_trace(go.Surface(x=plot_x, y=plot_y, z=Z), row=1, col=1)
fig.add_trace(go.Surface(x=plot_x, y=plot_y, z=Z_fitted), row=1, col=2)

fig.show()
```

{{< iframe figure_7.html 550 >}}


We can see this is working. We can also look at the marginal relationship for each input variable:


```python
plot_x_0 = np.linspace(
    params[0]["unique_vals"].min(),
    params[0]["unique_vals"].max(),
    len(params[0]["unique_vals"]),
)

fig = go.Figure(
    [
        go.Scatter(x=plot_x_0, y=np.sin(plot_x_0), name="True Function"),
        go.Scatter(
            x=params[0]["unique_vals"],
            y=params[0]["beta_vec"].value,
            name="Fitted Function",
        ),
    ],
)
fig.update_layout(title="Marginal Relationship for First Variable")
fig.show()
```

{{< iframe figure_8.html 550 >}}


```python
plot_x_1 = np.linspace(
    params[1]["unique_vals"].min(),
    params[1]["unique_vals"].max(),
    len(params[1]["unique_vals"]),
)

fig = go.Figure(
    [
        go.Scatter(x=plot_x_1, y=np.exp(plot_x_1), name="True Function"),
        go.Scatter(
            x=params[1]["unique_vals"],
            y=params[1]["beta_vec"].value,
            name="Fitted Function",
        ),
    ],
)
fig.update_layout(title="Marginal Relationship for Second Variable")
fig.show()
```


{{< iframe figure_9.html 550 >}}

We can see that for the first variable, the sine relationship is capture very well, but for the second variable, our fitted graph has the right shape but is shifted down. Let's look at the fitted intercept value:


```python
alpha.value
```




    array(6.17078177)



It's not 5 as we would expect. In fact, this is the identifiability problem rearing its head. If we take the difference between this $\alpha$ value and 5 we can see that it is very close to the gap between the true function value and our fitted curve for the second variable:


```python
# Average distance between true and fitted curve
print(np.mean(np.exp(plot_x_1) - params[1]["beta_vec"].value))
# Distance between fitted and true intercept values
print(alpha.value - 5)
```

    1.1728523581910208
    1.1707817719249132


The model has "moved" some of the intercept value to the marginal relationship for the second variable. Now, this has no effect when we look at our predictions, $\hat{y}$, because the effect naturally washes out, so our predicted vs actual surfaces above are very close. But we obviously don't recover the true intercept or the true marginal relationship for the second variable.

As far as I can tell, this is because the second variable follows an exponential curve. This means that the additive term for this variable will always be positive. You can see how this is an issue because we've constrained the model to have the total marginal effect sum to zero, when we expect the true total marginal effect to always be positive. If you replace the exponential function in the synthetic data code with something else that takes positive and negative values you can recover the correct intercept and marginal relationships. I have yet to work out if there is a better way to deal with this, or if some cases like the exponential are not fixable from this perspective. I'm not too worried about it because all of this only matters up to an additive constant, so the *shape* of the marginal relationship is correct, and predictions are unaffected, but it would be nice to be able to perfectly recover the true parameters in general. There may be a more clever way to handle the identifiability constraint that resolves this problem, but I don't know it.

# Input Standardization

One question that may arise with this model is whether we need to standardize our data beforehand like you have to in a normal ridge or lasso penalty setting. In those settings, if your model coefficients need to be on different scales the regularization penalty will improperly penalize larger variables more so than smaller ones, so you standardize the variables in advance so the penalty applied "equally" to everything.

My speculation is that it would make no difference here, although this is based on an argument that is very mathematically hand-wavy and I'm not confident in it. First, let's start with our assumption of the true model form:
$$
y_i = \alpha + f(x_{1, i}) + g(x_{2, i}) + \epsilon
$$
Our $\beta$ coefficients seek to estimate the values of the true functions at our sample points (we want our estimated functions $\hat{f}$ and $\hat{g}$ to be close to the true $f$ and $g$):
$$
\begin{align*}
\beta_{1, i} &= \hat{f}(x_{1, i}) \newline
\beta_{2, i} &= \hat{g}(x_{2, i})
\end{align*}
$$
Our penalty is then:
$$
\begin{align}
\vert \beta_{1, i+1} - \beta_{1, i} \vert &= \vert \hat{f}(x_{1, i+1}) - \hat{f}(x_{1, i}) \vert \newline
&= \vert \hat{f}(x_{1, i} + h) - \hat{f}(x_{1, i}) \vert \newline
&= \vert h \hat{f}'(x_{1, i}) \vert
\end{align}
$$
Where we get the second line by assuming that our input points are evenly spaced and close by and we get to the third line by using the usual limit definition of a derivative. You can easily make this same argument for $g$. Buying this, we can see that our penalty is based on the first derivative of our estimated function. This qualitatively means that larger regularization values will promote flatter estimated functions that converge to a constant function (a constant function has zero derivative), which is the behavior we see:


```python
lams = np.logspace(1, 3, 5)
betas = []
for lam in lams:
    # For each observed y value get the relevant beta coefficient for that X observation
    # by using the reconstruction index based on the unique values vector
    alpha = cp.Variable(name="alpha")
    y_hat = alpha + cp.sum(
        [params[i]["beta_vec"][params[i]["recon_idx"]] for i in params.keys()]
    )
    # Compute separate l1 norms for each input variable and sum
    penalty = cp.sum(
        [cp.norm(params[i]["D_mat"] @ params[i]["beta_vec"], 1) for i in params.keys()]
    )

    objective = cp.Minimize(0.5 * cp.sum_squares(obs_y - y_hat) + lam * penalty)
    # Sum to zero constraint to fix identifiability problems
    constraints = [cp.sum(params[i]["beta_vec"]) == 0 for i in params.keys()]
    prob = cp.Problem(objective, constraints)
    results = prob.solve(solver="CLARABEL")

    betas.append(params[0]["beta_vec"].value)
```


```python
fig = go.Figure(
    [
        go.Scatter(
            x=params[1]["unique_vals"],
            y=betas[i],
            name=f"Lambda Value: {lams[i]:.0f}",
        ) for i in range(len(lams))
    ],
)
fig.update_layout(title="First Variable Betas by Regularization Penalty")
fig.show()
```

{{< iframe figure_13.html 550 >}}

Note that our sum-to-zero constraint ensures that the function we fit converges to a constant zero. Moving back to our original problem, we can see from this:
$$
\vert h \hat{f}'(x_{1, i}) \vert
$$
that the only thing we will change by standardizing our $x$ values beforehand is the thing that goes *inside* the derivative of our fitted function. If we do this for our first and second variable, the relative magnitudes of the penalties will still depend on the exact form and magnitude of the first derivatives of our fitted functions, which should largely match with the true underlying function. In particular, given our first variable's true function is $sin(x)$ the derivative is $cos(x)$ so the magnitude of these values won't change at all whether we standardize or not:


```python
standard_X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
sort_idx = np.argsort(X[:, 0])

fig = go.Figure(
    [
        go.Scatter(x=X[:, 0][sort_idx], y=np.sin(X[:, 0][sort_idx]), name="Original Values"),
        go.Scatter(
            x=standard_X[:, 0][sort_idx],
            y=np.cos(standard_X[:, 0][sort_idx]),
            name="Standardized Values",
        ),
    ],
)
fig.update_layout(title="Original vs Transformed x Values for First Variable")
fig.show()
```

{{< iframe figure_14.html 550 >}}

So, we condense the range of $x$ values that are being supplied, but the magnitude of those values is the same. This is even more striking for our second variable because the derivative of the exponential is itself, so the values we get back are identical, but just evaluated over a smaller range of $x$ values.

All of this is to say that I don't think that standardizing the input values will help with the differing magnitude of penalty values across different variables. I do think that having a separate regularization term for each variable would be more correct but obviously this causes a more complicated hyper-parameter search scheme. At least for this problem, using a single regularization term seems to work, so I can somewhat confidently say that this issue is *less* significant in this setting than in the normal linear setting, but it certainly still matters. If you can afford the computation, you may get slightly better results giving each variable a separate regularization parameter.

# Conclusion

With this I think the basic mathematical and conceptual setup for extending trend filtering to GAMs is set. I still have some uncertainty about the need for applying separate penalty terms to each variable, but the general outline of how to make this work seems pretty clear. The only extension I would like to get around to eventually is covering a two-variate interaction case. You can imagine that instead of $\beta$ *vectors* you would end up with matrices and the penalty matrices would need another dimension.
