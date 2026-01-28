import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime as dt
from Dates import from_excel_date

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tick


# TODO: Check with Bauza re the validity of the results
def manual_var(annualized_volatility : pd.DataFrame, normalized_volatilities: pd.Series, positions: pd.Series, quantile : float = 1.96, custom_factors : pd.Series = None):
    """
    Bauza's manual risk monitor
    """
    securities = normalized_volatilities.index
    if custom_factors is None:
        custom_factors = pd.Series(np.ones(len(normalized_volatilities)), index = securities)
    portfolio_VaR = pd.DataFrame(index = securities)
    portfolio_VaR['Vol (Ann.)'] = annualized_volatility
    portfolio_VaR['Normalized'] = normalized_volatilities
    portfolio_VaR['Position'] = positions
    portfolio_VaR['Daily Vol (bps)'] = 100*portfolio_VaR.Normalized / np.sqrt(252)
    portfolio_VaR['Unit Variance'] = portfolio_VaR['Daily Vol (bps)']**2
    portfolio_VaR['VaR (97.5%)'] = abs(portfolio_VaR['Position']*portfolio_VaR['Daily Vol (bps)']*quantile*custom_factors)
    portfolio_VaR.loc['Total'] = portfolio_VaR.sum()
    portfolio_VaR['Risk Weights Squared'] = (portfolio_VaR['Position']*portfolio_VaR['Daily Vol (bps)'])**2
    return portfolio_VaR


def plot_timeseries(df: pd.DataFrame,
                    title: str = 'Time Series Plot',
                    xlabel : str = 'Date',
                    ylabel : str = 'Value'):
  """
  Plots a DataFrame with columns as time series and a legend in the upper right corner using seaborn.

  Args:
    df: pandas DataFrame with a datetime index and columns to plot.
  """
  sns.set(style="whitegrid")
  plt.figure(figsize=(12, 6))
  sns.lineplot(data=df, linestyle='-', dashes=False) # Set linestyle to solid
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend(loc='upper right')
  plt.tight_layout()
  plt.show()


def plot_with_confidence_interval(ts: pd.Series,
                                  lower_bound: pd.Series,
                                  upper_bound: pd.Series,
                                  title: str = '',
                                  label: str = '',
                                  xlabel: str = 'Date',
                                  ylabel: str = 'Value'):
  sns.set_style("whitegrid")
  plt.figure(figsize=(14, 6))
  plt.plot(ts.index, ts, label=label, color="red")
  plt.fill_between(ts.index, lower_bound, upper_bound, color="red", alpha=0.2, label="95% CI")
  plt.legend()
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend(loc='upper right')
  plt.tight_layout()
  plt.show()


def millions_formatter(x, pos):
    """The two args are the value and tick position."""
    return f'-${abs(x / 1_000_000):.2f}M'
formatter = tick.FuncFormatter(millions_formatter)


def parametric_var(portfolio: pd.Series,
                   alpha: float = 0.95,
                   notional: int = 10_000_000,
                   window : int = 100,
                   n_bootstrap: int = 250):
  """
  Computes the parametric VaR for a given portfolio at a level of alpha.
  """
  z_score = norm.ppf(1-alpha)
  mu = portfolio.rolling(window=window).mean().dropna()
  std = portfolio.rolling(window=window).std().dropna()

  upper_bound = pd.Series(index = portfolio.index)
  lower_bound = pd.Series(index = portfolio.index)

  for i, d in enumerate(portfolio.index):
    if i < window - 1:
      lower_bound[d] = np.nan
      lower_bound[d] = np.nan
      continue
    sliced = portfolio.iloc[i - window + 1:i + 1]
    resample = [np.random.choice(sliced, size=window, replace=True) for _ in range(n_bootstrap)]
    bootstrapped_mean = np.array([np.mean(x) for x in resample])
    bootstrapped_std = np.array([np.std(x) for x in resample])

    lower_bound[d]= np.percentile((bootstrapped_mean + z_score * bootstrapped_std), 2.5)
    upper_bound[d] = np.percentile((bootstrapped_mean + z_score * bootstrapped_std), 97.5)
  lower_bound = lower_bound.dropna()
  upper_bound = upper_bound.dropna()
  return (mu + z_score * std) * notional, lower_bound * notional, upper_bound * notional

def historical_var(portfolio: pd.Series,
                   alpha: float = 0.95,
                   notional : int = 10_000_000,
                   window :  int = 100,
                   n_bootstrap: int = 250):
  """
  Historical VaR with bootstrapped 95% CI
  """
  point_estimate = portfolio.rolling(window = window).quantile(1-alpha).dropna() * notional

  upper_bound = pd.Series(index = portfolio.index)
  lower_bound = pd.Series(index = portfolio.index)

  for i, d in enumerate(portfolio.index):
    if i < window - 1:
      continue
    sliced = portfolio.iloc[i - window + 1:i + 1]
    bootstrapped_quantiles = [
        np.percentile(np.random.choice(sliced, size=window, replace=True), (1 - alpha) * 100)
        for _ in range(n_bootstrap)
    ]
    lower_bound[d]= np.percentile(bootstrapped_quantiles, 2.5)
    upper_bound[d] = np.percentile(bootstrapped_quantiles, 97.5)
  return point_estimate, lower_bound.dropna() * notional, upper_bound.dropna() * notional


def parametric_marginal_var(returns: pd.DataFrame,
                            weights: pd.Series,
                            alpha: float = 0.95,
                            notional: int = 10_000_000,
                            window: int = 100, **kargs):
  """
  Computes the parametric marginal VaR for a given portfolio at a level of alpha.
  """
  z_score = norm.ppf(1-alpha)
  tickers = returns.columns
  marginal = pd.DataFrame(columns = tickers)
  for i in range(window, len(returns)):
    d = returns.index[i]
    sliced = returns.iloc[i+1-window:i+1]
    mu = sliced.mean()
    Sigma = np.cov(sliced.values, rowvar=False, ddof=1)
    sigma_p = float(np.sqrt(weights @ Sigma @ weights))
    marginal.loc[d] = (mu +  z_score * (Sigma.dot(weights) / sigma_p) )

  component = marginal * weights * notional

  if 'dates' in kargs:
    dates = kargs['dates']
    marginal = marginal.loc[dates]
    component = component.loc[dates]

  # Plot marginal and component VaR
  plot_timeseries(marginal, title='Marginal VaR per ticker', ylabel='Marginal VaR')
  plot_timeseries(component, title='Component VaR per ticker', ylabel='Component VaR')

  relative_component = component.apply(lambda x: x / x.sum(), axis=1)
  plt.figure(figsize=(15, 8))
  ax = plt.gca()
  colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 10))
  relative_component.plot(kind='bar', stacked=True, ax=ax, legend=False, color=colors, width=1.0, edgecolor = 'none')

  plt.title('Component VaR over time')
  plt.xlabel('Date')
  plt.ylabel('Component VaR (%)')

  handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(10)]
  labels = relative_component.columns.tolist()
  plt.legend(handles, labels, title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
  dates = relative_component.index
  tick_positions = np.arange(0, len(dates), step=max(1, int(len(dates)/10))) # Adjust step dynamically
  ax.set_xticks(tick_positions + 0.5) # Center ticks between bars
  ax.set_xticklabels([dates[i].strftime('%Y-%m') for i in tick_positions], rotation=45, ha='right')


  plt.grid(axis='y') # Only show horizontal grid lines for stacked bars
  plt.tight_layout()
  plt.show()

  diff = None
  if 'parametric' in kargs:
    parametric = kargs['parametric']
    diff = abs(component.sum(axis=1) - parametric)
    plot_timeseries(diff, title='Absolute difference Agg Component vs Parametric VaR (sanity check)', ylabel='Difference')

  return marginal, component, diff

def backtest(portfolio: pd.Series,
             VaR: float,
             window : int = 500):
  sliced = portfolio.iloc[-window:]
  counts = (sliced < VaR).value_counts()
  worst_diff = (sliced - VaR).min()
  return (counts.get(True, 0) / (counts.get(True, 0) + counts.get(False, 0))).item(), worst_diff