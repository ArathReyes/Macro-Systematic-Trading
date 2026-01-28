Macro Systematic
================

Research and backtesting toolkit for Macro rates and FX systematic strategies. It includes yield-curve PCA residual strategies, pairs trading tools, VaR/risk utilities, and interactive Dash dashboards for visualization.

Repository layout
-----------------
- `Data/` — Excel inputs (`Rates.xlsx`, `FX.xlsx`, `MX_Rates.xlsx`, `Treasuries.xlsx`, `FX Forwards.xlsx`). Sheet names and tenors are referenced from `config.json`.
- `Results/` — Pre-built HTML reports for backtests and stress tests by country plus Fly RV summaries.
- `RelativeValue.py` — PCA-based residual/mean-reversion engine with stress tools for buffers, windows, and signal persistence.
- `PairsTrading.py` & `pairs_dashboard_app.py` — Cointegration checks, spread dislocation tables, Bollinger plots, and a Dash dashboard for bonds/swaps/ASW in MX.
- `RiskManagement.py` & `VaR_app.py` — Parametric/historical VaR (with bootstrap CIs), marginal/component VaR, backtesting helpers, and a Dash VaR dashboard.
- `Plots.py`, `Stats.py`, `Dates.py`, `PCA.py`, `VolCarry.py` — Shared plotting, stationarity stats, date bumping, yield-curve PCA, and vol-carry straddle backtester utilities.
- Notebooks (`*.ipynb`) — Exploratory workflows for dispersion, ASW, fly RV, vol carry, etc.

Getting started
---------------
1) Install Python 3.10+ and create a virtual environment:
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
2) Install dependencies (adjust as needed):
```
pip install pandas numpy scipy statsmodels scikit-learn plotly dash seaborn matplotlib quantstats openpyxl
```
3) Place/update the required Excel inputs in `Data/` with the expected sheet names (see `config.json` for curve names and tenors).

Running the dashboards
----------------------
- Pairs trading: `python pairs_dashboard_app.py` (opens http://127.0.0.1:8050). Uses `Data/MX_Rates.xlsx` sheets `Swap` and `Bonds` to build ASW/bond/swap views.
- VaR: `python VaR_app.py` (opens http://127.0.0.1:8050). Uses `Data/FX.xlsx` (`Spot` sheet) to compute parametric/historical VaR with optional bootstrap CIs and Kupiec backtests.

Using the strategy modules
--------------------------
- Relative value: instantiate `RelativeValue(country="US", lookback="5Y")`, call `get_residuals()`, then `get_weights(...)` and `compute_pnl(...)` to produce PnL and index level series. Stress helpers: `stress_rebalacing`, `stress_windows`, `stress_signal_persistance`.
- Pairs: use `suitable_pairs(df)` for Engle-Granger cointegration + stationarity metrics; `plot_spreads_dislocations` and `plot_spread_bollinger` for visualization.
- Risk/VaR: `manual_var`, `parametric_var`, `historical_var`, `parametric_marginal_var`, and `backtest` provide standalone risk calculations; see `VaR_app.py` for integrated usage patterns.
- Vol carry: `VolCarryBacktester(data, tenor="1M").run()` simulates short straddle with optional delta hedging and transaction costs.

Notebooks and reports
---------------------
- `*.ipynb` notebooks capture exploratory analysis for ASW, dispersion, full RV, vol carry, and risk.
- HTML outputs under `Results/` and `Fly_RV_*.html` show saved dashboards/backtests for quick review.

Notes
-----
- Data files are not versioned here; supply your own market data with matching sheet/column names.
- Dash apps default to localhost; adjust host/port if deploying elsewhere.
