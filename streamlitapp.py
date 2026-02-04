import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Market Forecast App",
    page_icon="📈",
    layout="wide"
)

# =========================
# DARK UI
# =========================
st.markdown("""
<style>
.main { background-color: #020617; }
h1, h2, h3 { color: #38bdf8; }
.css-18e3th9, .css-1d391kg { background-color: #020617; }
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("📊 Stock, Index & Commodity Forecast Dashboard")

# =========================
# SAFE CLOSE COLUMN HANDLER
# =========================
def normalize_close_column(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    return df.dropna()

# =========================
# DATA SOURCES
# =========================
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

nifty50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
    "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS"
]

# =========================
# SIDEBAR
# =========================
selection = st.sidebar.radio(
    "📌 Data Source",
    ["Indices", "Nifty 50 Stocks", "Gold & Silver", "Upload CSV"]
)

ticker = None
stock_data = None

# =========================
# LOAD DATA
# =========================
if selection == "Indices":
    name = st.sidebar.selectbox("Select Index", indices.keys())
    ticker = indices[name]
    stock_data = yf.download(ticker, period="1y")

elif selection == "Nifty 50 Stocks":
    ticker = st.sidebar.selectbox("Select Stock", nifty50)
    stock_data = yf.download(ticker, period="1y")

elif selection == "Gold & Silver":
    name = st.sidebar.selectbox("Select Commodity", commodities.keys())
    ticker = commodities[name]
    stock_data = yf.download(ticker, period="1y")

elif selection == "Upload CSV":
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file:
        stock_data = pd.read_csv(file, index_col=0, parse_dates=True)
        ticker = "Uploaded CSV"
        if 'Close' not in stock_data.columns:
            st.error("CSV must contain a 'Close' column")
            stock_data = None

# =========================
# MAIN LOGIC
# =========================
if stock_data is not None and not stock_data.empty:

    stock_data = normalize_close_column(stock_data)

    # =========================
    # METRIC
    # =========================
    latest_price = stock_data['Close'].iloc[-1]
    st.metric("💰 Latest Price", f"{latest_price:,.2f}")

    # =========================
    # TABLE
    # =========================
    st.subheader("📄 Market Data")
    st.dataframe(
        stock_data[['Open','High','Low','Close']],
        use_container_width=True
    )

    # =========================
    # LIVE CHART
    # =========================
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data['Close'],
        name="Close Price",
        line=dict(color="#38bdf8")
    ))
    fig.update_layout(
        title=f"📈 Live Price - {ticker}",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # FORECAST MODELS
    # =========================
    def arima_forecast(data, steps=1260):  # 5 years
        model = ARIMA(data['Close'], order=(5,1,0))
        fit = model.fit()
        forecast = fit.forecast(steps)
        idx = pd.date_range(data.index[-1], periods=steps+1, freq='B')[1:]
        return pd.DataFrame({"Forecast": forecast}, index=idx)

    def sarima_forecast(data, steps=1260):  # FIXED SARIMA
        model = SARIMAX(
            data['Close'],
            order=(1,1,1),
            seasonal_order=(0,1,1,252),  # yearly seasonality
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fit = model.fit(disp=False)
        forecast = fit.get_forecast(steps)
        df = forecast.summary_frame()

        df = df[['mean','mean_ci_lower','mean_ci_upper']]
        df.columns = ['Forecast','Lower','Upper']

        idx = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=steps,
            freq='B'
        )
        df.index = idx
        return df

    # =========================
    # FORECAST UI
    # =========================
    st.sidebar.subheader("🔮 Forecast")
    model_choice = st.sidebar.radio("Model", ["ARIMA", "SARIMA"])

    if st.sidebar.button("🚀 Generate Forecast"):

        forecast_df = (
            arima_forecast(stock_data)
            if model_choice == "ARIMA"
            else sarima_forecast(stock_data)
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['Close'],
            name="Historical",
            line=dict(color="#22c55e")
        ))
        fig2.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Forecast'],
            name="Forecast",
            line=dict(color="#f97316")
        ))

        if "Lower" in forecast_df.columns:
            fig2.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['Lower'],
                name="Lower Bound",
                line=dict(dash="dash")
            ))
            fig2.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['Upper'],
                name="Upper Bound",
                line=dict(dash="dash")
            ))

        fig2.update_layout(
            title=f"📉 {ticker} 5-Year Forecast ({model_choice})",
            template="plotly_dark",
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👈 Select a data source from sidebar to begin")
