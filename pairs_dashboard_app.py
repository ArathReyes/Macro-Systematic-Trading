import os
import numpy as np
import pandas as pd
from pathlib import Path

import dash
from dash import dcc, html, dash_table, Input, Output, State, no_update

from PairsTrading import plot_spreads_dislocations, plot_spread_bollinger
from PairsTrading import suitable_pairs

import webbrowser
from threading import Timer


# Map user-friendly instrument names to data locations.
DATA_PATHS = {
    "Bond": Path("Data/Treasuries.xlsx"),
    "Swap": Path("Data/Rates.xlsx"),
    "ASW": Path("Data/MX_Rates.xlsx"),
}


def _load_pairs_data(instrument: str) -> pd.DataFrame:
    """
    Best-effort loader for the selected instrument. Falls back to a synthetic sample
    so the dashboard can render even if files are missing.
    """

    swap = pd.read_excel('data/MX_Rates.xlsx', index_col=0, sheet_name='Swap', parse_dates=True)
    swap.index = [d.date() for d in swap.index]
    bonds = pd.read_excel('data/MX_Rates.xlsx', index_col=0, sheet_name='Bonds', parse_dates=True)
    bonds.index = [d.date() for d in bonds.index]
    tenors = swap.columns.intersection(bonds.columns)
    ASW = (bonds[tenors] - swap[tenors])*100
    ASW.dropna(inplace = True, how = 'all')
    if instrument.upper() == 'ASW':
        return ASW
    elif instrument.upper() == 'BOND':
        return bonds
    elif instrument.upper() == 'SWAP':
        return swap
    else:
        raise Exception(f'Instrument ({instrument}) invalid. Only available for bonds, swaps and ASW.')

def _make_empty_table(message: str):
    return pd.DataFrame({"Info": [message]})


def _blank_fig(message: str):
    return {
        "data": [],
        "layout": {
            "template": "plotly_white",
            "annotations": [{"text": message, "showarrow": False, "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5}],
        },
    }


app = dash.Dash(__name__, title="Pairs Dashboard", suppress_callback_exceptions=True)

app.layout = html.Div(
    [
        html.H2("Pairs Trading Dashboard", style={"margin": "0", "fontWeight": "700"}),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Instrument", style={"fontWeight": "600"}),
                                dcc.Dropdown(
                                    id="instrument-select",
                                    options=[{"label": k, "value": k} for k in DATA_PATHS.keys()],
                                    value="Bond",
                                    clearable=False,
                                ),
                                html.Label("Stationarity Lookback (rows)", style={"fontWeight": "600", "marginTop": "10px"}),
                                dcc.Input(
                                    id="stationarity-lookback",
                                    type="number",
                                    value=252,
                                    min=30,
                                    step=10,
                                    style={"width": "100%"},
                                ),
                            ],
                            style={"padding": "12px", "background": "#edf2ff", "borderRadius": "10px"},
                        ),
                        html.Div(
                            [
                                html.H4("Stationarity Table", style={"marginBottom": "6px"}),
                                dash_table.DataTable(
                                    id="stationarity-table",
                                    style_table={"maxHeight": "360px", "overflowY": "auto"},
                                    style_header={"fontWeight": "700", "backgroundColor": "#f4f6fb"},
                                    style_cell={"textAlign": "center", "padding": "6px"},
                                ),
                            ],
                            style={"marginTop": "12px"},
                        ),
                    ],
                    style={"display": "flex", "flexDirection": "column", "gap": "12px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Z-Score Window", style={"fontWeight": "600"}),
                                dcc.Input(id="zscore-window", type="number", value=63, min=20, step=5, style={"width": "100%"}),
                                html.Label("Confidence", style={"fontWeight": "600", "marginTop": "10px"}),
                                dcc.Slider(
                                    id="confidence-slider",
                                    min=0.5,
                                    max=0.99,
                                    step=0.01,
                                    value=0.75,
                                    marks=None,
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                            ],
                            style={"padding": "12px", "background": "#e8f6f3", "borderRadius": "10px"},
                        ),
                        html.Div(
                            [
                                html.H4("Spread Z-Score Table", style={"marginBottom": "6px"}),
                                dcc.Graph(id="dislocation-table", style={"height": "360px"}),
                            ],
                            style={"marginTop": "12px"},
                        ),
                    ],
                    style={"display": "flex", "flexDirection": "column", "gap": "12px"},
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "16px",
                "alignItems": "start",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Long Tenor", style={"fontWeight": "600"}),
                        dcc.Dropdown(id="long-tenor", clearable=False),
                    ],
                    style={"flex": 1, "minWidth": "180px"},
                ),
                html.Div(
                    [
                        html.Label("Short Tenor", style={"fontWeight": "600"}),
                        dcc.Dropdown(id="short-tenor", clearable=False),
                    ],
                    style={"flex": 1, "minWidth": "180px"},
                ),
                html.Div(
                    [
                        html.Label("Sigma Factor", style={"fontWeight": "600"}),
                        dcc.Slider(
                            id="sigma-factor",
                            min=0.5,
                            max=3.5,
                            step=0.1,
                            value=2.0,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    style={"flex": 1, "minWidth": "220px"},
                ),
            ],
            style={
                "display": "flex",
                "gap": "12px",
                "marginTop": "18px",
                "padding": "12px",
                "background": "#fff7ed",
                "borderRadius": "10px",
            },
        ),
        html.Div(
            [
                html.H4("Spread Bollinger Bands", style={"marginBottom": "6px"}),
                dcc.Graph(id="bollinger-plot", style={"height": "520px"}),
            ],
            style={"marginTop": "12px"},
        ),
    ],
    style={
        "padding": "18px",
        "fontFamily": "IBM Plex Sans, Arial",
        "background": "#f7f9fc",
        "color": "#1f2933",
        "display": "flex",
        "flexDirection": "column",
        "gap": "12px",
    },
)


@app.callback(
    Output("stationarity-table", "data"),
    Output("stationarity-table", "columns"),
    Output("long-tenor", "options"),
    Output("short-tenor", "options"),
    Output("long-tenor", "value"),
    Output("short-tenor", "value"),
    Input("instrument-select", "value"),
    Input("stationarity-lookback", "value"),
)
def update_stationarity_table(instrument, lookback):
    data = _load_pairs_data(instrument)
    if data.empty:
        empty = _make_empty_table("No data available.")
        columns = [{"name": c, "id": c} for c in empty.columns]
        return empty.to_dict("records"), columns, [], [], None, None

    lookback = int(lookback or len(data))
    sliced = data.tail(lookback)
    try:
        res = suitable_pairs(sliced)
        res = res[(res['Cointegrated'])].sort_values('Half-Life (Days)')
        res_display = res.reset_index().rename(columns={"index": "Pair"})
    except Exception as exc:
        res_display = _make_empty_table(f"Error: {exc}")

    columns = [{"name": col, "id": col} for col in res_display.columns]
    long_short_options = [{"label": c, "value": c} for c in data.columns]
    default_long = data.columns[0] if len(data.columns) > 0 else None
    default_short = data.columns[1] if len(data.columns) > 1 else default_long

    return res_display.to_dict("records"), columns, long_short_options, long_short_options, default_long, default_short


@app.callback(
    Output("dislocation-table", "figure"),
    Input("instrument-select", "value"),
    Input("zscore-window", "value"),
    Input("confidence-slider", "value"),
)
def update_dislocation_table(instrument, window, confidence):
    data = _load_pairs_data(instrument)
    if data.empty:
        return _blank_fig("No data for z-score table.")

    window = int(window or 63)
    confidence = float(confidence or 0.75)
    sliced = data.tail(max(window, 2))
    try:
        fig = plot_spreads_dislocations(sliced, window=window, confidence=confidence)
    except Exception as exc:
        return _blank_fig(f"Error building dislocation table: {exc}")
    return fig


@app.callback(
    Output("bollinger-plot", "figure"),
    Input("instrument-select", "value"),
    Input("long-tenor", "value"),
    Input("short-tenor", "value"),
    Input("sigma-factor", "value"),
    Input("zscore-window", "value"),
)
def update_bollinger_plot(instrument, long_tenor, short_tenor, sigma_factor, window):
    data = _load_pairs_data(instrument)
    if data.empty or not long_tenor or not short_tenor:
        return _blank_fig("Select tenors to view the Bollinger plot.")

    window = int(window or 63)
    sigma_factor = float(sigma_factor or 2.0)
    try:
        fig = plot_spread_bollinger(
            data=data,
            long_tenor=long_tenor,
            short_tenor=short_tenor,
            window=window,
            sigma_factor=sigma_factor,
        )
    except Exception as exc:
        return _blank_fig(f"Error building Bollinger plot: {exc}")
    return fig



def open_browser():
    # Prevents opening multiple tabs when the Werkzeug reloader runs
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new("http://127.0.0.1:8050/") # Or the port you specify


if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True, host='127.0.0.1', port=8050)