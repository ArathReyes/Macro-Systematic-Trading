import os
import dash
from dash import dcc, html, dash_table, Input, Output, State, ALL
import pandas as pd
import numpy as np
from scipy.stats import norm, chi2
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import warnings
import webbrowser
from threading import Timer

from Dates import from_excel_date

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'default_alpha': 0.95,
    'default_window': 252,
    'default_notional': 10_000_000,
    'default_bootstrap': 100,
    'backtest_window': 500,
    'random_seed': 42
}

np.random.seed(CONFIG['random_seed'])

# ==========================================
# DATA GENERATION (Replace with your loader)
# ==========================================
def generate_dummy_data():
    spot = pd.read_excel('Data/FX.xlsx', sheet_name='Spot', index_col=0)
    spot.drop('Date', inplace = True)
    spot.index.rename = 'Date'
    spot.index = spot.index.map(from_excel_date)
    spot.dropna(inplace = True)

    returns = np.log((spot / spot.shift(1)).astype(float)).dropna()
    
    return returns, spot

# Load data
returns, spot = generate_dummy_data()

# ==========================================
# CALCULATION FUNCTIONS
# ==========================================

def validate_inputs(alpha, window, notional, n_bootstrap, data_length):
    """Validate user inputs"""
    errors = []
    
    if not (0.01 <= alpha <= 0.99):
        errors.append("Alpha must be between 0.01 and 0.99")
    
    if window < 10:
        errors.append("Window must be at least 10 days")
    
    if window > data_length:
        errors.append(f"Window ({window}) exceeds available data ({data_length} days)")
    
    if notional <= 0:
        errors.append("Notional must be positive")
    
    if n_bootstrap < 0:
        errors.append("Bootstrap iterations must be non-negative")
    
    return errors

def calculate_portfolio_returns(returns_df, weights_dict):
    """Calculate portfolio returns from weights"""
    weights = pd.Series(weights_dict)
    # Normalize weights
    weights = weights / weights.sum()
    return returns_df[weights.index].dot(weights)

def parametric_var_calc(portfolio, alpha, notional, window, n_bootstrap=None):
    """
    Calculate parametric VaR with optional bootstrap confidence intervals
    
    Parameters:
    -----------
    portfolio : pd.Series - Portfolio returns
    alpha : float - Confidence level
    notional : int - Portfolio notional value
    window : int - Rolling window size
    n_bootstrap : int - Number of bootstrap samples (None to skip CI)
    
    Returns:
    --------
    result : pd.Series - VaR estimates
    lower_bound : pd.Series - Lower CI bound (or None)
    upper_bound : pd.Series - Upper CI bound (or None)
    """
    z_score = norm.ppf(1 - alpha)
    mu = portfolio.rolling(window=window).mean()
    std = portfolio.rolling(window=window).std()
    
    result = (mu + z_score * std) * notional
    
    if n_bootstrap is None or n_bootstrap == 0:
        return result.dropna(), None, None
    
    # Bootstrap confidence intervals
    upper_bound = pd.Series(index=portfolio.index, dtype=float)
    lower_bound = pd.Series(index=portfolio.index, dtype=float)
    
    for i in range(window - 1, len(portfolio)):
        d = portfolio.index[i]
        sliced = portfolio.iloc[i - window + 1:i + 1]
        
        # Vectorized bootstrapping
        resample = np.random.choice(sliced.values, size=(n_bootstrap, window), replace=True)
        bootstrapped_mean = np.mean(resample, axis=1)
        bootstrapped_std = np.std(resample, axis=1, ddof=1)
        
        boot_var = bootstrapped_mean + z_score * bootstrapped_std
        lower_bound[d] = np.percentile(boot_var, 2.5)
        upper_bound[d] = np.percentile(boot_var, 97.5)
    
    return result.dropna(), lower_bound.dropna() * notional, upper_bound.dropna() * notional

def historical_var_calc(portfolio, alpha, notional, window, n_bootstrap=None):
    """Calculate historical VaR with optional bootstrap CI"""
    point_estimate = portfolio.rolling(window=window).quantile(1 - alpha) * notional
    
    if n_bootstrap is None or n_bootstrap == 0:
        return point_estimate.dropna(), None, None
    
    upper_bound = pd.Series(index=portfolio.index, dtype=float)
    lower_bound = pd.Series(index=portfolio.index, dtype=float)
    
    for i in range(window - 1, len(portfolio)):
        d = portfolio.index[i]
        sliced = portfolio.iloc[i - window + 1:i + 1].values
        
        resample = np.random.choice(sliced, size=(n_bootstrap, window), replace=True)
        bootstrapped_quantiles = np.percentile(resample, (1 - alpha) * 100, axis=1)
        
        lower_bound[d] = np.percentile(bootstrapped_quantiles, 2.5)
        upper_bound[d] = np.percentile(bootstrapped_quantiles, 97.5)
    
    return point_estimate.dropna(), lower_bound.dropna() * notional, upper_bound.dropna() * notional

def parametric_marginal_var_calc(returns_df, weights, alpha, notional, window):
    """Calculate marginal and component VaR"""
    z_score = norm.ppf(1 - alpha)
    tickers = returns_df.columns
    marginal = pd.DataFrame(columns=tickers, index=returns_df.index[window:])
    
    for i in range(window, len(returns_df)):
        d = returns_df.index[i]
        sliced = returns_df.iloc[i + 1 - window:i + 1]
        mu = sliced.mean()
        Sigma = np.cov(sliced.values, rowvar=False, ddof=1)
        sigma_p = float(np.sqrt(weights @ Sigma @ weights))
        
        if sigma_p == 0:
            marginal.loc[d] = 0
        else:
            marginal.loc[d] = (mu + z_score * (Sigma @ weights / sigma_p))
    
    component = marginal.multiply(weights) * notional
    return marginal, component

def kupiec_test(exceptions, total_observations, alpha):
    """
    Kupiec POF (Proportion of Failures) test
    
    Returns:
    --------
    lr_stat : float - Likelihood ratio test statistic
    p_value : float - P-value
    result : str - Test result interpretation
    """
    p = 1 - alpha  # Expected exception rate
    if exceptions == 0:
        lr_stat = 2 * total_observations * np.log(1 / (1 - p))
    elif exceptions == total_observations:
        lr_stat = 2 * total_observations * np.log(1 / p)
    else:
        p_hat = exceptions / total_observations
        lr_stat = -2 * (
            total_observations * (p * np.log(p_hat) + (1 - p) * np.log(1 - p_hat))
            - total_observations * (p * np.log(p) + (1 - p) * np.log(1 - p))
        )
    
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    
    # Traffic light interpretation
    if p_value >= 0.05:
        result = "GREEN"  # Model is adequate
    elif p_value >= 0.01:
        result = "YELLOW"  # Model is questionable
    else:
        result = "RED"  # Model is inadequate
    
    return lr_stat, p_value, result

def backtest_var(portfolio, var_series, window, notional, alpha):
    """
    Comprehensive VaR backtesting
    
    Returns:
    --------
    dict with backtest statistics
    """
    # Align data
    portfolio_pnl = (portfolio * notional).iloc[-window:]
    var_aligned = var_series.iloc[-window:]
    
    # Find exceptions (where loss exceeds VaR)
    exceptions = (portfolio_pnl < var_aligned).sum()
    total_obs = len(portfolio_pnl)
    exception_rate = exceptions / total_obs
    expected_rate = 1 - alpha
    
    # Kupiec test
    lr_stat, p_value, traffic_light = kupiec_test(exceptions, total_obs, alpha)
    
    # Find worst breach
    breaches = portfolio_pnl - var_aligned
    worst_breach = breaches.min()
    
    # Average VaR
    avg_var = var_aligned.mean()
    
    return {
        'exceptions': exceptions,
        'total_obs': total_obs,
        'exception_rate': exception_rate,
        'expected_rate': expected_rate,
        'lr_stat': lr_stat,
        'p_value': p_value,
        'traffic_light': traffic_light,
        'worst_breach': worst_breach,
        'avg_var': avg_var
    }

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

def create_var_plot(portfolio, var_series, lower, upper, title, alpha, start_date=None, end_date=None):
    """Enhanced VaR plot with P&L overlay and breaches"""
    fig = go.Figure()
    
    # --- FIX 1: FILTERING LOGIC (Previously fixed) ---
    if start_date and end_date:
        mask = (var_series.index >= start_date) & (var_series.index <= end_date)
        var_series = var_series[mask]
        
        if lower is not None:
            lower = lower[mask]
        if upper is not None:
            upper = upper[mask]
            
        # Align portfolio to the filtered VaR index
        portfolio = portfolio.loc[var_series.index]
    
    # --- FIX 2: CONCATENATION LOGIC (New fix) ---
    if lower is not None and upper is not None:
        # Use .append() for Index objects, not pd.concat()
        ci_x = lower.index.append(upper.index[::-1])
        
        # pd.concat() is fine for Series objects (the y-values)
        ci_y = pd.concat([lower, upper[::-1]])
        
        fig.add_trace(go.Scatter(
            x=ci_x,
            y=ci_y,
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='95% Bootstrap CI',
            hoverinfo='skip'
        ))
    
    # VaR Line
    fig.add_trace(go.Scatter(
        x=var_series.index,
        y=var_series,
        mode='lines',
        line=dict(color='red', width=2),
        name=f'VaR ({int(alpha*100)}%)'
    ))
    
    # Actual P&L (aligned with VaR dates)
    pnl = portfolio * 10_000_000
    fig.add_trace(go.Scatter(
        x=pnl.index,
        y=pnl,
        mode='lines',
        line=dict(color='blue', width=1),
        name='Daily P&L',
        opacity=0.7
    ))
    
    # Mark VaR breaches
    breaches = pnl < var_series
    if breaches.any():
        fig.add_trace(go.Scatter(
            x=pnl.index[breaches],
            y=pnl[breaches],
            mode='markers',
            marker=dict(color='red', size=8, symbol='x'),
            name='VaR Breaches',
            hovertemplate='<b>Breach</b><br>Date: %{x}<br>P&L: $%{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value ($)",
        template="plotly_white",
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig

def create_component_chart(component_df, start_date=None, end_date=None):
    """Create interactive component VaR stacked area chart"""
    if start_date and end_date:
        mask = (component_df.index >= start_date) & (component_df.index <= end_date)
        component_df = component_df[mask]
    
    # Sample data if too many points
    if len(component_df) > 500:
        component_df = component_df.iloc[::len(component_df)//500]
    
    fig = go.Figure()
    
    for col in component_df.columns:
        fig.add_trace(go.Scatter(
            x=component_df.index,
            y=component_df[col],
            mode='lines',
            stackgroup='one',
            name=col,
            hovertemplate='%{y:$,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Component VaR Attribution Over Time",
        xaxis_title="Date",
        yaxis_title="VaR Contribution ($)",
        template="plotly_white",
        hovermode='x unified',
        legend_title="Currency Pair"
    )
    
    return fig

def create_distribution_plot(portfolio, var_value, alpha):
    """Create P&L distribution histogram with VaR overlay"""
    fig = go.Figure()
    
    # Histogram of returns
    fig.add_trace(go.Histogram(
        x=portfolio * 10_000_000,
        nbinsx=50,
        name='P&L Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # VaR line
    fig.add_vline(
        x=var_value,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR ({int(alpha*100)}%)",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Portfolio P&L Distribution",
        xaxis_title="Daily P&L ($)",
        yaxis_title="Frequency",
        template="plotly_white",
        showlegend=True
    )
    
    return fig

# ==========================================
# DASH APP
# ==========================================

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.Div([
        html.H1("VaR Portfolio Analysis", style={'margin': '0', 'color': 'white'}),
        html.P("Advanced Risk Analytics with Backtesting & Component Analysis", 
               style={'margin': '5px 0 0 0', 'color': 'rgba(255,255,255,0.8)'})
    ], style={
        'backgroundColor': '#2c3e50',
        'padding': '20px',
        'marginBottom': '20px',
        'borderRadius': '5px'
    }),
    
    # Control Panel
    html.Div([
        html.H3("Control Panel", style={'marginTop': '0'}),
        
        html.Div([
            # Left Column
            html.Div([
                html.Label("Confidence Level (α)", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='input-alpha',
                    min=0.90,
                    max=0.99,
                    step=0.01,
                    value=CONFIG['default_alpha'],
                    marks={0.90: '90%', 0.95: '95%', 0.99: '99%'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                
                html.Label("Rolling Window (Days)", style={'fontWeight': 'bold', 'marginTop': '15px'}),
                dcc.Input(
                    id='input-window',
                    type='number',
                    value=CONFIG['default_window'],
                    min=10,
                    step=1,
                    style={'width': '100%'}
                ),
                
                html.Label("Portfolio Notional ($)", style={'fontWeight': 'bold', 'marginTop': '15px'}),
                dcc.Input(
                    id='input-notional',
                    type='number',
                    value=CONFIG['default_notional'],
                    min=1000,
                    step=100000,
                    style={'width': '100%'}
                ),
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
            
            # Middle Column
            html.Div([
                html.Label("Bootstrap Iterations", style={'fontWeight': 'bold'}),
                dcc.Input(
                    id='input-bootstrap',
                    type='number',
                    value=CONFIG['default_bootstrap'],
                    min=0,
                    step=50,
                    style={'width': '100%'}
                ),
                
                html.Label("Backtest Window (Days)", style={'fontWeight': 'bold', 'marginTop': '15px'}),
                dcc.Input(
                    id='input-backtest-window',
                    type='number',
                    value=CONFIG['backtest_window'],
                    min=50,
                    step=50,
                    style={'width': '100%'}
                ),
                
                html.Div([
                    dcc.Checklist(
                        id='include-ci',
                        options=[{'label': ' Show Confidence Intervals', 'value': 'ci'}],
                        value=['ci'],
                        style={'marginTop': '15px'}
                    )
                ])
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
            
            # Right Column - Date Range
            html.Div([
                html.Label("Analysis Date Range", style={'fontWeight': 'bold'}),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=returns.index[0],
                    end_date=returns.index[-1],
                    display_format='YYYY-MM-DD',
                    style={'width': '100%'}
                ),
                
                html.Button(
                    'Run Analysis',
                    id='btn-run',
                    n_clicks=0,
                    style={
                        'backgroundColor': '#27ae60',
                        'color': 'white',
                        'padding': '12px 30px',
                        'border': 'none',
                        'cursor': 'pointer',
                        'fontSize': '16px',
                        'fontWeight': 'bold',
                        'borderRadius': '5px',
                        'marginTop': '20px',
                        'width': '100%'
                    }
                ),
            ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
        ]),
    ], style={
        'backgroundColor': '#ecf0f1',
        'padding': '20px',
        'borderRadius': '5px',
        'marginBottom': '20px'
    }),
    
    # Error Display
    html.Div(id='error-display'),
    
    # Loading Spinner
    dcc.Loading(
        id="loading",
        type="cube",
        children=[
            # Results Container
            html.Div(id='results-container', style={'display': 'none'}, children=[
                
                # Summary Cards
                html.Div(id='summary-cards', style={'marginBottom': '20px'}),
                
                # VaR Charts
                html.Div([
                    html.Div([
                        dcc.Graph(id='graph-parametric')
                    ], style={'width': '49%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(id='graph-historical')
                    ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
                ]),
                
                html.Hr(),
                
                # Distribution
                html.Div([
                    dcc.Graph(id='graph-distribution')
                ], style={'marginBottom': '20px'}),
                
                html.Hr(),
                
                # Component VaR
                html.Div([
                    dcc.Graph(id='graph-component')
                ]),
                
                html.Hr(),
                
                # Backtest Results
                html.H3("Backtest Results", style={'textAlign': 'center'}),
                html.Div([
                    dash_table.DataTable(
                        id='table-backtest',
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px',
                            'fontFamily': 'Arial'
                        },
                        style_header={
                            'backgroundColor': '#34495e',
                            'color': 'white',
                            'fontWeight': 'bold'
                        },
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'traffic_light', 'filter_query': '{traffic_light} = GREEN'},
                                'backgroundColor': '#d4edda',
                                'color': '#155724'
                            },
                            {
                                'if': {'column_id': 'traffic_light', 'filter_query': '{traffic_light} = YELLOW'},
                                'backgroundColor': '#fff3cd',
                                'color': '#856404'
                            },
                            {
                                'if': {'column_id': 'traffic_light', 'filter_query': '{traffic_light} = RED'},
                                'backgroundColor': '#f8d7da',
                                'color': '#721c24'
                            },
                        ],
                        data=[]
                    )
                ], style={'width': '95%', 'margin': '0 auto'})
            ])
        ]
    ),
    
    # Footer
    html.Div([
        html.P("© 2024 VaR Analytics Dashboard | Data refreshed: " + datetime.now().strftime("%Y-%m-%d %H:%M"),
               style={'textAlign': 'center', 'color': '#7f8c8d', 'margin': '0'})
    ], style={'marginTop': '40px', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px'})
])

# ==========================================
# CALLBACKS
# ==========================================

@app.callback(
    [Output('error-display', 'children'),
     Output('results-container', 'style'),
     Output('summary-cards', 'children'),
     Output('graph-parametric', 'figure'),
     Output('graph-historical', 'figure'),
     Output('graph-distribution', 'figure'),
     Output('graph-component', 'figure'),
     Output('table-backtest', 'data'),
     Output('table-backtest', 'columns')],
    [Input('btn-run', 'n_clicks')],
    [State('input-alpha', 'value'),
     State('input-window', 'value'),
     State('input-notional', 'value'),
     State('input-bootstrap', 'value'),
     State('input-backtest-window', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('include-ci', 'value')]
)
def update_dashboard(n_clicks, alpha, window, notional, n_bootstrap, bt_window, 
                     start_date, end_date, include_ci):
    
    if n_clicks == 0:
        return None, {'display': 'none'}, None, go.Figure(), go.Figure(), go.Figure(), go.Figure(), [], []
    
    try:
        # Validate inputs
        errors = validate_inputs(alpha, window, notional, n_bootstrap, len(returns))
        if errors:
            error_div = html.Div([
                html.H4("⚠️ Input Validation Errors:", style={'color': '#e74c3c'}),
                html.Ul([html.Li(error) for error in errors])
            ], style={
                'backgroundColor': '#f8d7da',
                'padding': '15px',
                'borderRadius': '5px',
                'border': '1px solid #f5c6cb'
            })
            return error_div, {'display': 'none'}, None, go.Figure(), go.Figure(), go.Figure(), go.Figure(), [], []
        
        # Calculate portfolio returns (equal weights for demo)
        N = len(returns.columns)
        weights = pd.Series([1/N]*N, index=returns.columns)
        portfolio = returns.dot(weights)
        
        # Determine if CI should be calculated
        n_boot = n_bootstrap if ('ci' in include_ci and include_ci) else 0
        
        # Calculate VaR
        param_var, param_low, param_up = parametric_var_calc(portfolio, alpha, notional, window, n_boot)
        hist_var, hist_low, hist_up = historical_var_calc(portfolio, alpha, notional, window, n_boot)
        
        # Component VaR
        marginal, component = parametric_marginal_var_calc(returns, weights, alpha, notional, window)
        
        # Backtest
        param_bt = backtest_var(portfolio, param_var, bt_window, notional, alpha)
        hist_bt = backtest_var(portfolio, hist_var, bt_window, notional, alpha)
        
        # Summary Cards
        current_param_var = param_var.iloc[-1]
        current_hist_var = hist_var.iloc[-1]
        
        summary_cards = html.Div([
            # 1. Parametric VaR Card (Existing)
            html.Div([
                html.H4("Current Parametric VaR", style={'margin': '0', 'color': '#7f8c8d'}),
                html.H2(f"${current_param_var:,.0f}", style={'margin': '10px 0', 'color': '#e74c3c'}),
                html.P(f"{int(alpha*100)}% confidence, {window}-day window", style={'margin': '0', 'fontSize': '12px'})
            ], style={
                'width': '23%',
                'display': 'inline-block',
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '5px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'margin': '0 1%',
                'verticalAlign': 'top'
            }),

            # 2. Historical VaR Card (NEW)
            html.Div([
                html.H4("Current Historical VaR", style={'margin': '0', 'color': '#7f8c8d'}),
                html.H2(f"${current_hist_var:,.0f}", style={'margin': '10px 0', 'color': '#3498db'}), # Blue color
                html.P(f"{int(alpha*100)}% confidence, {window}-day window", style={'margin': '0', 'fontSize': '12px'})
            ], style={
                'width': '23%',
                'display': 'inline-block',
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '5px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'margin': '0 1%',
                'verticalAlign': 'top'
            })
        ])
        
        # Create plots
        fig_param = create_var_plot(
            portfolio, param_var, param_low, param_up,
            f'Parametric VaR ({window}-day Rolling Window)',
            alpha, start_date, end_date
        )
        
        fig_hist = create_var_plot(
            portfolio, hist_var, hist_low, hist_up,
            f'Historical VaR ({window}-day Rolling Window)',
            alpha, start_date, end_date
        )
        
        fig_dist = create_distribution_plot(
            portfolio, current_param_var, alpha
        )
        
        fig_comp = create_component_chart(component, start_date, end_date)
        
        # Backtest table
        backtest_columns = [
            {"name": "Model", "id": "model"},
            {"name": "Current VaR", "id": "current_var"},
            {"name": "Avg VaR", "id": "avg_var"},
            {"name": "Exceptions", "id": "exceptions"},
            {"name": "Exception Rate", "id": "exception_rate"},
            {"name": "Expected Rate", "id": "expected_rate"},
            {"name": "LR Statistic", "id": "lr_stat"},
            {"name": "P-Value", "id": "p_value"},
            {"name": "Traffic Light", "id": "traffic_light"},
            {"name": "Worst Breach", "id": "worst_breach"}
        ]
        
        backtest_data = [
            {
                'model': f'Parametric ({window}d)',
                'current_var': f'${current_param_var:,.0f}',
                'avg_var': f'${param_bt["avg_var"]:,.0f}',
                'exceptions': f'{param_bt["exceptions"]}/{param_bt["total_obs"]}',
                'exception_rate': f'{param_bt["exception_rate"]*100:.2f}%',
                'expected_rate': f'{param_bt["expected_rate"]*100:.2f}%',
                'lr_stat': f'{param_bt["lr_stat"]:.2f}',
                'p_value': f'{param_bt["p_value"]:.4f}',
                'traffic_light': param_bt['traffic_light'],
                'worst_breach': f'${param_bt["worst_breach"]:,.0f}'
            },
            {
                'model': f'Historical ({window}d)',
                'current_var': f'${current_hist_var:,.0f}',
                'avg_var': f'${hist_bt["avg_var"]:,.0f}',
                'exceptions': f'{hist_bt["exceptions"]}/{hist_bt["total_obs"]}',
                'exception_rate': f'{hist_bt["exception_rate"]*100:.2f}%',
                'expected_rate': f'{hist_bt["expected_rate"]*100:.2f}%',
                'lr_stat': f'{hist_bt["lr_stat"]:.2f}',
                'p_value': f'{hist_bt["p_value"]:.4f}',
                'traffic_light': hist_bt['traffic_light'],
                'worst_breach': f'${hist_bt["worst_breach"]:,.0f}'
            }
        ]
        
        return (None, {'display': 'block'}, summary_cards, 
                fig_param, fig_hist, fig_dist, fig_comp, 
                backtest_data, backtest_columns)
    
    except Exception as e:
        error_div = html.Div([
            html.H4("❌ Calculation Error:", style={'color': '#e74c3c'}),
            html.P(str(e), style={'fontFamily': 'monospace', 'backgroundColor': '#f8f9fa', 'padding': '10px'}),
            html.P("Please check your inputs and try again.", style={'marginTop': '10px'})
        ], style={
            'backgroundColor': '#f8d7da',
            'padding': '20px',
            'borderRadius': '5px',
            'border': '1px solid #f5c6cb',
            'marginBottom': '20px'
        })
        
        return error_div, {'display': 'none'}, None, go.Figure(), go.Figure(), go.Figure(), go.Figure(), [], []

# ==========================================
# RUN APP
# ==========================================

def open_browser():
    # Prevents opening multiple tabs when the Werkzeug reloader runs
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new("http://127.0.0.1:8050/") # Or the port you specify



if __name__ == '__main__':
    print("=" * 60)
    print("Starting Enhanced VaR Dashboard...")
    print("=" * 60)
    print("Access the dashboard at: http://127.0.0.1:8050")
    print("Press CTRL+C to stop the server")
    print("=" * 60)
    Timer(1, open_browser).start()
    app.run(debug=True, host='127.0.0.1', port=8050)