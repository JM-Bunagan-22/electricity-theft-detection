"""
Interactive dashboard for reviewing theft-risk-ranked customers: a
sortable table of the highest-risk accounts, a consumption chart for the
selected customer, and a feature-importance / model-metrics summary so a
reviewer can see why an account was flagged.
"""

import json
import os

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCORED = pd.read_csv(os.path.join(DATA_DIR, "scored_customers.csv"))
LONG_DF = pd.read_csv(os.path.join(DATA_DIR, "consumption_long.csv"), parse_dates=["date"])
with open(os.path.join(DATA_DIR, "metrics.json")) as f:
    METRICS = json.load(f)

TOP_N = 200
TOP_CUSTOMERS = SCORED.head(TOP_N).copy()
TOP_CUSTOMERS["risk_score"] = TOP_CUSTOMERS["risk_score"].round(3)
TOP_CUSTOMERS["outcome"] = TOP_CUSTOMERS["flag"].map({1: "Confirmed theft", 0: "Normal"})

app = Dash(__name__)
app.title = "Electricity Theft Risk Dashboard"

STAT_CARDS = [
    ("Customers scored", f"{METRICS['n_customers']:,}"),
    ("Confirmed theft cases", f"{METRICS['n_theft']:,} ({METRICS['theft_rate']:.1%})"),
    ("Random Forest ROC-AUC", f"{METRICS['random_forest']['roc_auc']:.3f}"),
    ("Random Forest PR-AUC", f"{METRICS['random_forest']['pr_auc']:.3f}"),
    ("Isolation Forest ROC-AUC", f"{METRICS['isolation_forest']['roc_auc']:.3f}"),
    ("Recall @ best F1 threshold", f"{METRICS['random_forest']['recall_theft']:.1%}"),
]

importance_fig = go.Figure(
    go.Bar(
        x=list(METRICS["feature_importance"].values())[::-1],
        y=list(METRICS["feature_importance"].keys())[::-1],
        orientation="h",
        marker_color="#2563eb",
    )
)
importance_fig.update_layout(
    title="Random Forest feature importance",
    margin=dict(l=140, r=20, t=40, b=30),
    height=340,
    plot_bgcolor="white",
)

app.layout = html.Div(
    style={"fontFamily": "Inter, Arial, sans-serif", "padding": "24px", "maxWidth": "1200px", "margin": "0 auto"},
    children=[
        html.H1("⚡ Electricity Theft Risk Dashboard", style={"marginBottom": "4px"}),
        html.P(
            "Random Forest risk scores vs. confirmed SGCC theft labels — top 200 highest-risk customers.",
            style={"color": "#555", "marginTop": 0},
        ),
        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "margin": "20px 0"},
            children=[
                html.Div(
                    style={
                        "background": "#f4f6fb",
                        "borderRadius": "10px",
                        "padding": "14px 18px",
                        "minWidth": "170px",
                        "flex": "1",
                    },
                    children=[
                        html.Div(label, style={"fontSize": "12px", "color": "#666", "textTransform": "uppercase"}),
                        html.Div(value, style={"fontSize": "22px", "fontWeight": "700", "color": "#1f2937"}),
                    ],
                )
                for label, value in STAT_CARDS
            ],
        ),
        html.Div(
            style={"display": "flex", "gap": "24px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"flex": "1.3", "minWidth": "480px"},
                    children=[
                        html.H3("Top risk-ranked customers"),
                        dash_table.DataTable(
                            id="risk-table",
                            columns=[
                                {"name": "Customer ID", "id": "customer_id"},
                                {"name": "Risk score", "id": "risk_score"},
                                {"name": "Actual outcome", "id": "outcome"},
                            ],
                            data=TOP_CUSTOMERS[["customer_id", "risk_score", "outcome"]].to_dict("records"),
                            row_selectable="single",
                            selected_rows=[0],
                            page_size=12,
                            sort_action="native",
                            style_cell={"padding": "8px", "fontSize": "13px"},
                            style_header={"fontWeight": "700", "background": "#f4f6fb"},
                            style_data_conditional=[
                                {
                                    "if": {"filter_query": '{outcome} = "Confirmed theft"'},
                                    "backgroundColor": "#fef2f2",
                                    "color": "#b91c1c",
                                }
                            ],
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": "1", "minWidth": "420px"},
                    children=[dcc.Graph(figure=importance_fig)],
                ),
            ],
        ),
        html.H3("Daily consumption for selected customer", style={"marginTop": "28px"}),
        dcc.Graph(id="consumption-chart"),
    ],
)


@app.callback(
    Output("consumption-chart", "figure"),
    Input("risk-table", "selected_rows"),
)
def update_chart(selected_rows):
    if not selected_rows:
        customer_id = TOP_CUSTOMERS.iloc[0]["customer_id"]
    else:
        customer_id = TOP_CUSTOMERS.iloc[selected_rows[0]]["customer_id"]

    hist = LONG_DF[LONG_DF["customer_id"] == customer_id].sort_values("date")
    row = SCORED[SCORED["customer_id"] == customer_id].iloc[0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist["date"], y=hist["kwh"], mode="lines", name="Daily kWh",
            line=dict(color="#2563eb", width=1.2),
        )
    )
    outcome = "Confirmed theft" if row["flag"] == 1 else "Normal"
    fig.update_layout(
        title=f"Customer {customer_id} — risk score {row['risk_score']:.3f} — {outcome}",
        margin=dict(l=40, r=20, t=50, b=30),
        height=380,
        plot_bgcolor="white",
        xaxis_title="Date",
        yaxis_title="kWh / day",
    )
    return fig


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
