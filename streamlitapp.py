import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Indian Market Intelligence",
    page_icon="🇮🇳",
    layout="wide"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>
h1 {
    background: linear-gradient(90deg,#f97316,#22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
section[data-testid="stSidebar"] {
    background-color: #020617;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def normalize(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

def future_dates(last_date, steps):
    return pd.bdate_range(last_date + pd.offsets.BDay(1), periods=steps)

def forecast_with_band(df, steps):
    prices = df['Close'].values
    returns = np.diff(prices) / prices[:-1]
    mean, vol = returns.mean(), returns.std()

    price = prices[-1]
    f, u, l = [], [], []

    for _ in range(steps):
        price *= (1 + mean)
        band = price * vol * 2
        f.append(price)
        u.append(price + band)
        l.append(price - band)

    idx = future_dates(df.index[-1], steps)
    return pd.DataFrame({"Forecast": f, "Upper": u, "Lower": l}, index=idx)

def market_trend(df):
    ma50 = df['Close'].rolling(50).mean()
    if df['Close'].iloc[-1] > ma50.iloc[-1]:
        return "🟢 BULLISH"
    elif df['Close'].iloc[-1] < ma50.iloc[-1]:
        return "🔴 BEARISH"
    return "🟡 SIDEWAYS"

# =========================
# DATA LISTS
# =========================
banks = {
    "SBI": "SBIN.NS",
    "HDFC": "HDFCBANK.NS",
    "ICICI": "ICICIBANK.NS",
    "Axis": "AXISBANK.NS",
    "Kotak": "KOTAKBANK.NS",
    "PNB": "PNB.NS",
    "Bank of Baroda": "BANKBARODA.NS",
    "Canara": "CANBK.NS"
}

indices = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^NSEIT"
}

commodities = {
    "Gold": "GC=F",
    "Silver": "SI=F"
}

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📌 Market Control")

mode = st.sidebar.radio(
    "Mode",
    ["Single Market Forecast", "Bank-wise Comparison", "Upload CSV"]
)

days = st.sidebar.slider("Forecast Days", 30, 300, 120)

# =========================
# SINGLE MARKET MODE
# =========================
if mode == "Single Market Forecast":

    group = st.sidebar.selectbox(
        "Category",
        ["Indices", "Banks", "Commodities"]
    )

    if group == "Indices":
        name = st.sidebar.selectbox("Select", indices.keys())
        ticker = indices[name]
    elif group == "Banks":
        name = st.sidebar.selectbox("Select", banks.keys())
        ticker = banks[name]
    else:
        name = st.sidebar.selectbox("Select", commodities.keys())
        ticker = commodities[name]

    run = st.sidebar.button("🚀 Generate Forecast")

    df = yf.download(ticker, period="5y", progress=False)
    df = normalize(df)

    st.title(f"{name} Dashboard")
    st.subheader(f"Trend: {market_trend(df)}")

    # ---- TABLE ----
    st.subheader("📄 Market Data")
    st.dataframe(df[['Open','High','Low','Close']].tail(200),
                 use_container_width=True)

    forecast = forecast_with_band(df, days) if run else None

    # ---- CHART ----
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Historical"))

    if forecast is not None:
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Forecast'],
                                 name="Forecast", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Upper'],
                                 showlegend=False, line=dict(width=0)))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Lower'],
                                 fill='tonexty', name="Range",
                                 fillcolor='rgba(34,197,94,0.25)',
                                 line=dict(width=0)))

    fig.update_layout(
        template="plotly_dark",
        height=550,
        font=dict(family="Inter, Arial, sans-serif", color="#cbd5e1"),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

# =========================
# BANK COMPARISON
# =========================
elif mode == "Bank-wise Comparison":

    st.title("🏦 Bank-wise Comparison & Heatmap")

    data = {}
    for name, tick in banks.items():
        dfb = yf.download(tick, period="1y", progress=False)
        if not dfb.empty:
            data[name] = dfb['Close'].pct_change().sum() * 100

    heat_df = pd.DataFrame.from_dict(
        data, orient='index', columns=['1Y % Change']
    )

    fig = go.Figure(data=go.Heatmap(
        z=heat_df.values,
        y=heat_df.index,
        x=heat_df.columns,
        colorscale='RdYlGn'
    ))
    fig.update_layout(
        height=400,
        template="plotly_dark",
        font=dict(family="Inter, Arial, sans-serif", color="#cbd5e1"),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

    st.subheader("📄 Bank Performance Table")
    st.dataframe(heat_df.style.format("{:.2f}%"))

# =========================
# CSV MODE
# =========================
else:
    st.title("📂 CSV Forecast")

    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if not file:
        st.stop()

    df = pd.read_csv(file)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    df = normalize(df)

    st.subheader(f"Trend: {market_trend(df)}")

    # ---- TABLE ----
    st.subheader("📄 Uploaded Data")
    cols = [c for c in ['Open','High','Low','Close'] if c in df.columns]
    st.dataframe(df[cols].tail(200), use_container_width=True)

    forecast = forecast_with_band(df, days)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Historical"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Forecast'],
                             name="Forecast", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Upper'],
                             showlegend=False, line=dict(width=0)))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Lower'],
                             fill='tonexty', name="Range",
                             fillcolor='rgba(249,115,22,0.25)',
                             line=dict(width=0)))

    fig.update_layout(
        template="plotly_dark",
        height=550,
        font=dict(family="Inter, Arial, sans-serif", color="#cbd5e1"),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})
