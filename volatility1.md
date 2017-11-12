### Heteroscedasticity

> Heteroscedasticity is a hard word to pronounce, but it doesn't need to be a difficult concept to understand. Put simply, heteroscedasticity (also spelled heteroskedasticity) refers to the circumstance in which the variability of a variable is unequal across the range of values of a second variable that predicts it.

> A scatterplot of these variables will often create a cone-like shape, as the scatter (or variability) of the dependent variable (DV) widens or narrows as the value of the independent variable (IV) increases. The inverse of heteroscedasticity is homoscedasticity, which indicates that a DV's variability is equal across values of an IV. 
> (see reference [heteroscedasticity](http://www.statsmakemecry.com/smmctheblog/confusing-stats-terms-explained-heteroscedasticity-heteroske.html "heteroscedasticity"))

- We use the **Apple** stock price for a demonstration.
- Below is the formula for daily log return for index spot levels
  $ \displaystyle Z_t = \rm{ln}~\frac{P_t}{P_{t-1}} $
- Below figure is Apple 5 year historical spot
  ![Apple 5Y Spot History](https://image.ibb.co/jeDvoG/apple5Y.png "Apple 5Y Spot Data")
- There is also daily log return for **Apple**
  ![Apple 5Y Daily Log Return](http://image.ibb.co/dGKMvw/apple5_Ydaily_LR.png "Apple 5Y Daily Log Return")
- For the above figures, they can be generated with Pandas in python

```python
>>> import pandas_datareader as pdr
>>> import numpy as np
>>> import datetime
>>> from dateutil.relativedelta import relativedelta
>>> end = datetime.date.today()
>>> from dateutil.relativedelta import relativedelta
>>> apple = pdr.DataReader( 'AAPL', 'yahoo', end-relativedelta(years=5), end )
>>> t = [ datetime.datetime.fromtimestamp( v/1e9 ) for v in apple.index.values.tolist() ]
>>> price = apple.Close.values.tolist()
>>> dchange = np.log( [ v/price[ k-1 ] for k, v in enumerate( price ) if k>0 ] )
```

First we use **AR(p)** model to fit the daily log return series $\{z_t\}$ to extract shocks.

> Given a time series $\{y_t\}$, the Autoregressive, i.e. *AR(p)* , process, aims to model the conditional expectation $\rm{E}(~y_t|\mathscr{F_{t-1}}~)$ with lagged dependent variables. 
> The **AR(p)** model is defined as
> $\displaystyle X_t = c + \sum_{i=1}^p~\phi_i X_{t-i} + \epsilon_t$
> where $\phi_1, \dots, \phi_p$ are model parameters, $\epsilon_t$ is white noise and $c$ is a constant.
> See reference [Stationary, AR(p), MA(p)](https://www.bauer.uh.edu/rsusmel/phd/ec2-3.pdf "Stationarity, AR(p), MA(p)")

Before we proceed, we also need [Bayesian Information Criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion "BIC Criterion") to help determine if the daily log return time series $\{z_t\}$ has **AR**, i.e. autoregression, structure. Also we need [Akaike Information Criterion](https://en.wikipedia.org/wiki/Akaike_information_criterion "AIC Criterion") to determine **AR** coefficient.

If the underlying process follows **AR** model, there will be a certain $p$ beyond which the *Autocorrelation Function* (**ACF**), which is defined as $\rho_k = \rm{Corr}( y_t, y_{t-t} )$, starts to either decay exponentially (**AR(p)** model), or directly cuts off to become $0$ (**MA(p)** model).

Note that the **ACF** is defined as
$\rho_k = \rm{Corr}(y_t, y_{t-k}) = \frac{\rm{Cov}(y_t, y_{t-k})}{\rm{Var}(y_t)} = \frac{\rm{E}((y_t-\mu)\cdot(y_{t-k}-\mu))}{\rm{Var}(y_t)}$
where $\mu$ is the sample mean.

In general, for time series analysis on a given time series $\{y_t\}$, we wish to decide on a model to use. In order to do that, we start with analyzing auto-correlation:

- Auto-correlation Correlogram. 

> Seasonal patterns of time series can be examined via correlograms, with **ACF** calculated numerically and displayed graphically. In *pandas*, the plotting tools and *statsmodels* graphics standardize/normalize the data before calculating the auto-correlation, in other words, they de-mean the data and divide by the standard deviation.
>
> As we mentioned earlier above, when normalizing the data, it is implicily assumed that the data is underlied by a Gaussian distribution, which is far from the truth generally.

We import first the necessary modules

```python
>>> from scipy import stats
>>> import matplotlib.pyplot as plt
>>> import statsmodels.api as sm
```

> And we can use statsmodels graphics functions to visualize the autocorrelation in the spot price time series and the daily log return time series

```python
>>> fig = plt.figure()
>>> ax1 = fig.add_subplot(211)
>>> sm.graphics.tsa.plot_acf( price, lags = 60, ax=ax1)
>>> ax2 = fig.add_subplot(212)
>>> sm.graphics.tsa.plot_acf( dchange, lags=60, ax=ax2)
>>> ax1.set_title( 'Apple 5Y Spot Autocorrelation' )
>>> ax2.set_title( 'Apple 5Y Daily Log Return Autocorrelation' )
```

![Apple 5Y autocorrelation](http://image.ibb.co/huRF5w/apple5_Yacf.png "Apple 5Y autocorrelation")

> By default the *statsmodels.graphics.tsa.plot_acf* sets the $95\%$ confidence interval. Any autocorrelation falling outside of the cone is very likely ***not*** a result of statistical coincidence. 
>
> As we can observe, the price time series have very heavy autocorrelation, but the daily log return time series seems to be uncorrelated between current and lagged data points.

- Partial Autocorrelation Function (**PACF**)

> This is an extention of autocorrelation, with the dependence on the intermediate elements ( those within the lag ) removed. So, in **PACF** only autocorrelation between current and single lagged data points is considered.

With both results from **ACF** and **PACF**, we apply below rule of thumbs to summarize our findings:

1. If **ACF** has exponential decay and **PACF** has a spike at lag $1$ with no correlation for other lags, then we pick one autoregressive (p) parameter.
2. If **ACF** shows a sine-wave ( sinuoidal ) pattern or maybe a series of exponential decays, and **PACF** has spikes at lags $1$ and $2$ with no correlation for other lags, then we use two autoregressive (p) parameters.
3. If **ACF** has spikes at lags $1$ and $2$ with no correlation for other lags, and the **PACF** having sine-wave ( sinuoidal ) patterns or a series of exponential decays, then we use two *Moving Average* (q) parameters.
4. If **ACF** shows exponential decay from lag $1$, and **PACF** shows exponential decay from lag $1$, then we use one autoregressive (p) and one *Moving Average* (q) parameters.

Consider **PACF** plots now for the Apple spot and daily log return time series.

```python
>>> fig = plt.figure()
>>> ax1 = fig.add_subplot(211)
>>> sm.
<matplotlib.figure.Figure object at 0x7f443f600650>graphics.tsa.plot_pacf( price, lags=60, ax=ax1 )
>>> ax2 = fig.add_subplot(212)
>>> sm.graphics.tsa.plot_pacf( dchange, lags=60, ax=ax2 )
>>> ax1.set_title( 'Apple 5Y Spot Partial Autocorrelation' )
>>> ax2.set_title( 'Apple 5Y Daily Log Return Partial Autocorrelation' )
```

![Apple 5Y spot and daily log return paritial autocorrelation](http://image.ibb.co/d6gMdG/apple5_Ypacf.png "Apple 5Y Spot and Dail Logreturn PACF")

From the above we could argue that the price dynamics can be modeled by a **AR(2)** process while the daily log return shows no autoregressive structure.

- Removing Serial Dependency

> There are two primary reasons for serial dependency removal
>
> - We can identify hidden nature of seasonal dependencies in the time series. Autocorrelations for consecutive lags are inter-dependent, so removal of some ( $1$ or $2$ ) of the autocorrelations will change the behavior of other inter-dependent autocorrelations, thus rendering seasonality pattern more apparent.
> - Removing serial dependency is necessary for **ARIMA** and other techniques by makng the time series stationary.

- Durbin-Watson Statistics

> This is another tool to infer/reflect serial dependency/correlation. 
>
> - A score of $2$ or value nearby indicates no first order serial correlation.
> - A score higher than $2$ indicates negative serial correlation.
> - A score below $2$ indicates positive serial correlation.

```python
>>> sm.stats.durbin_watson( price )
0.00020611229112895053
>>> sm.stats.durbin_watson( dchange )
1.9269710279008647
```

As suspected, the price data for Apple has very high positive serial correlation, and close to none first order serial correlation exists for the daily log return time series.

- We can also check out the **autocorrelation_plot** function from *pandas* 

```python
>>> from pandas.tools.plotting import autocorrelation_plot
>>> pricets = ( price - np.mean( price ) ) / np.std( price )
>>> plt.acorr( pricets, maxlags = len( pricets ) - 1, linestyle = 'solid', usevlines = False, marker = '' )
>>> autocorrelation_plot( pricets )
```

![Apple 5Y Spot autocorrelation using matplotlib.pyplot.acorr](http://image.ibb.co/eOLmCb/apple5_Ypltacorr.png "Apple 5Y Spot ACF using plotting acorr") ![Apple 5Y Spot autocorrelation using pandas plotting autocorrelation_plot](http://image.ibb.co/g3szXb/apple5_Ypdplottingacorr.png "Apple 5Y Spot pandas plotting autocorrelation_plot")

Now we can do the same thing for daily log return time series

![Apple 5Y daily log return time series autocorrelation](http://image.ibb.co/j2tjyG/apple5_YLRmatacorr.png)![Apple 5Y daily log return time series autocorrelation](http://image.ibb.co/n07BdG/apple5_YLRpdplottingacorr.png)

Before moving on to measure **BIC** and **AIC**, here we spend some time discussing two types of time series models:

- ARMA model. This model has no differencing terms. For this model to work, first the time series needs to be normalized. On subset of this ARMA model ( $q= 0$) is called autoregressive, i.e. **AR**, model. Its coefficient $p$ tells people how many lagged past values are included. The simplest **AR** models are AR(1) and AR(2) models, where $y_t = a_1 y_{t-1} + \epsilon_t$ is AR(1) and $y_t = a_1y_{t-1} + a_2y_{t-2} + \epsilon_t$ is AR(2). Here $\epsilon_t$ is also called **noise, error, random shock or residual**. 
- ARIMA model. This is a nonseasonal model, with coefficients $p, d, q$.  This model 
  - $p$ is the number of autoregressive terms
  - $q$ is the number of lagged forecast errors in the prediction equation
  - $d$ is the number of nonseasonal differences needed for stationarity

With the above discusion we are ready to test out if Apple 5Y time series can be fit with **AR** model.

```python
>>> for i in range(1, 10):
>>>     ar = sm.tsa.ARMA( pricets, (i,0)).fit()
>>>     ts += [ (ar.bic, ar.aic) ]
>>> ts_bic = [ v[0] for v in ts ]
>>> ts_aic = [ v[1] for v in ts ]
>>> ts_bic
[-3683.9288817901738, -3678.104158860021, -3672.2616868366035, -3665.7462773627503, -3657.4673501739048, -3644.2091150706792, -3637.5476756509215, -3635.8776569718625, -3638.1377064231947]
>>> ts_aic
[-3699.3454827900105, -3698.659626859803, -3697.9560218363313, -3696.5794793624236, -3693.4394191735237, -3685.3200510702436, -3683.797478650431, -3687.2663269713175, -3694.6652434225953]
>>> from pprint import pprint
>>> para = []
>>> for i in range(1, 10):
...     ar = sm.tsa.ARMA( pricets, (i,0)).fit(disp=0)
...     para += [ ar.params ]
>>> print para
[array([ 0.5391567 ,  0.99931284]), array([ 0.52480202,  1.03158641, -0.0323095 ]), array([ 0.53900419,  1.03260334, -0.06537156,  0.03208206]), array([ 0.54761278,  1.03188954, -0.06393619,  0.00912625,  0.02225809]), array([ 0.55570571,  1.02101326, -0.07248766,  0.01356156,  0.07540423,
       -0.03830892]), array([  6.63836303e+01,   1.03241101e+00,  -6.34912888e-02,
         8.09853617e-03,   3.99125120e-02,  -6.03634186e-03,
        -1.08947797e-02]), array([  6.93082732e+01,   1.03250086e+00,  -6.31901076e-02,
         7.17718799e-03,   3.98007739e-02,  -4.81459471e-03,
        -3.27698218e-02,   2.12953802e-02]), array([  6.41537409e+01,   1.03389658e+00,  -6.54572005e-02,
         6.96818414e-03,   4.24811788e-02,  -4.44036367e-03,
        -3.66387200e-02,   8.85853766e-02,  -6.53954434e-02]), array([ 0.530215  ,  1.03617857, -0.06852668,  0.00821132,  0.04192651,
       -0.00532303, -0.03716906,  0.09101347, -0.10263509,  0.03561377])]
```

Both **AIC** and **BIC** are pretty flat across all lags from $1$ to $10$. The coefficients of lagged terms are all very small.

Now with the daily log return time series, we see that

```python
>>> for i in range(1, 10):
>>>     ar = sm.tsa.ARMA( dchangets, (i,0)).fit()
>>>     ts += [ (ar.bic, ar.aic) ]
>>> ts_bic = [ v[0] for v in ts ]
>>> ts_aic = [ v[1] for v in ts ]
>>> ts_bic = [ v[0] for v in ts ]
>>> ts_aic = [ v[1] for v in ts ]
>>> ts_bic
[3592.7810821078388, 3598.5002915583509, 3605.2573920134569, 3612.0740267872316, 3618.1609580250242, 3625.2937675843446, 3625.9346132411874, 3627.2298576466219, 3634.309597689964]
>>> ts_aic
[3577.366863005706, 3577.9479994221733, 3579.567026843235, 3581.2455885829654, 3582.194446786714, 3584.18918331199, 3579.6919559347884, 3575.8491273061786, 3577.790794315476]
>>> para = []
>>> for i in range(1, 10):
...     ar = sm.tsa.ARMA( dchangets, (i,0)).fit(disp=0)
...     para += [ ar.params ]
... 
>>> pprint( para )
[array([ -1.48668166e-05,   3.47295188e-02]),
 array([  2.40012798e-05,   3.58934053e-02,  -3.35455796e-02]),
 array([  5.55316281e-05,   3.53114722e-02,  -3.29353100e-02,
        -1.73939838e-02]),
 array([  3.12918391e-05,   3.55889838e-02,  -3.24133092e-02,
        -1.79483489e-02,   1.59715050e-02]),
 array([ -3.80750456e-05,   3.60672606e-02,  -3.28614221e-02,
        -1.87490103e-02,   1.69640087e-02,  -2.90948686e-02]),
 array([ -4.41927422e-05,   3.60089117e-02,  -3.28258150e-02,
        -1.87855133e-02,   1.69061012e-02,  -2.90198039e-02,
        -2.06106150e-03]),
 array([ 0.00022074,  0.03614531, -0.03079878, -0.02003435,  0.0181938 ,
       -0.02692235, -0.00466375,  0.07229583]),
 array([ -4.87561568e-05,   4.10468249e-02,  -3.10751608e-02,
        -2.16972696e-02,   1.94739717e-02,  -2.84732812e-02,
        -6.39515925e-03,   7.47787834e-02,  -6.85913087e-02]),
 array([ -8.23967015e-06,   4.15094277e-02,  -3.15902346e-02,
        -2.16655310e-02,   1.96512688e-02,  -2.85608621e-02,
        -6.22698844e-03,   7.49583817e-02,  -6.88659820e-02,
         6.87112377e-03])]
```

It's the same for daily log return time series. We see that the residual/error term ( the first one in each case ) is very small and the coefficients are all small.

While lower values of AIC indicate that a model is more likely to be **close to** the truth, and lower values of BIC suggests a model is more likely to be the true, we find that for all the lag values tested, none of them seem to be standing out to be closer to true model ( **AIC, BIC** values are pretty flat). 

We thus can infer that the price time series of Apple has significant trend that persists for more than 60 days and, since the daily log return time series appear to be uncorrelated, the daily log returns are subject to models not covered by autoregression (**AR**).