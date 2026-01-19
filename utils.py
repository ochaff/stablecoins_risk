import pandas as pd
import plotly.graph_objects as go
import numpy as np

def plot_crv():
    df = pd.read_parquet('./data/curve_3pool_hourly')
    x = df.index

    w_dai  = df["w_DAI"].to_numpy()
    w_usdt = df["w_USDT"].to_numpy()
    w_usdc = df["w_USDC"].to_numpy()

    cum1 = w_dai
    cum2 = w_dai + w_usdt
    cum3 = w_dai + w_usdt + w_usdc  

    fig = go.Figure()


    fig.add_trace(go.Scatter(
        x=x, y=cum1,
        mode="lines",
        line=dict(width=0),
        fill="tozeroy",
        fillcolor="rgba(220,20,60,0.7)",  # crimson w/ alpha
        name="DAI",
        hovertemplate="%{x}<br>DAI: %{customdata:.2%}<extra></extra>",
        customdata=w_dai
    ))


    fig.add_trace(go.Scatter(
        x=x, y=cum2,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(255,165,0,0.7)",  # orange w/ alpha
        name="USDT",
        hovertemplate="%{x}<br>USDT: %{customdata:.2%}<extra></extra>",
        customdata=w_usdt
    ))


    fig.add_trace(go.Scatter(
        x=x, y=cum3,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(65,105,225,0.7)",  # royalblue w/ alpha
        name="USDC",
        hovertemplate="%{x}<br>USDC: %{customdata:.2%}<extra></extra>",
        customdata=w_usdc
    ))

    fig.update_layout(
        width=1000,
        height=550,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            x=0.7, y=-0.15,
            xanchor="center",
            yanchor="top"
        ),
        margin=dict(l=60, r=20, t=20, b=80),
    )


    ticks = np.linspace(0, 1, 6)
    fig.update_yaxes(
        range=[0, 1],
        tickmode="array",
        tickvals=ticks,
        ticktext=[f"{int(v*100)} %" for v in ticks],
        title=None
    )
    fig.update_xaxes(title=None)
    fig.write_html("./figures/crv_balance.html")
    fig.show()