import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Define available indices and stock tickers
indices = {
    'Nifty 50': '^NSEI',
    'Sensex': '^BSESN',
    'Nifty Bank': '^NSEBANK',
    'Nifty IT': '^NSEIT',
    'S&P BSE Small Cap': '^BSESMLCAP'
}

nifty50_tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS', 'INFY.NS',
    'HDFC.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'LT.NS', 'ITC.NS',
    'SBIN.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'NTPC.NS',
    'HCLTECH.NS', 'M&M.NS', 'ULTRACEMCO.NS', 'ONGC.NS', 'POWERGRID.NS',
    'SUNPHARMA.NS', 'TATAMOTORS.NS', 'WIPRO.NS', 'HDFC LIFE.NS', 'TATACONSUM.NS',
    'DIVISLAB.NS', 'TECHM.NS', 'DRREDDY.NS', 'SHREECEM.NS', 'JSW STEEL.NS',
    'BHARTIARTL.NS', 'ADANIGREEN.NS', 'GAIL.NS', 'CIPLA.NS', 'IOC.NS',
    'HEROMOTOCO.NS', 'TATAPOWER.NS', 'MUTHOOTFIN.NS', 'SBI LIFE.NS', 'BAJAJ AUTO.NS',
    'LUPIN.NS', 'TATASTEEL.NS', 'SAIL.NS', 'ICICI PRU.NS', 'AMBUJACEM.NS'
]

# Streamlit app
st.title('Indian Stock Market Indices Forecast')

# Initialize ticker variable
ticker = None

# Sidebar for selecting index or stock
selection = st.sidebar.radio("Select Data Source", ["Indices", "Nifty 50 Stocks", "Upload CSV"])

if selection == "Indices":
    selected_index = st.sidebar.selectbox("Select Index", options=list(indices.keys()))
    ticker = indices[selected_index]
    stock_data = yf.download(ticker, period='1y')
elif selection == "Nifty 50 Stocks":
    selected_stock = st.sidebar.selectbox("Select Stock", options=nifty50_tickers)
    ticker = selected_stock
    stock_data = yf.download(ticker, period='1y')
elif selection == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        stock_data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        st.sidebar.write(f"CSV file uploaded with {len(stock_data)} rows.")
        st.write(stock_data.head())  # Display the first few rows of the uploaded file
    else:
        st.sidebar.write("Please upload a CSV file.")
        stock_data = None

if stock_data is not None and not stock_data.empty:
    # Calculate daily profit and loss
    stock_data['Profit/Loss'] = stock_data['Close'].diff()
    stock_data['Profit/Loss Color'] = np.where(stock_data['Profit/Loss'] > 0, 'green', 'red')

    # Display ticker information
    if ticker is None:
        ticker = "Uploaded CSV Data"

    # Show live data
    st.subheader(f'Live Data for {ticker}')
    st.write(stock_data[['Open', 'High', 'Low', 'Close']])

    # Plot live data with plotly
    def plot_live_data(data):
        fig = go.Figure()

        # Add traces for Open, High, Low, and Close prices
        fig.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Open Price', line=dict(color='cyan', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index, y=data['High'], mode='lines', name='High Price', line=dict(color='green', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low Price', line=dict(color='red', dash='dash')))

        # Add profit/loss indicators
        for i in range(1, len(data)):
            if data['Profit/Loss'].iloc[i] > 0:
                fig.add_trace(go.Scatter(x=[data.index[i]], y=[data['Close'].iloc[i]], mode='markers+text',
                                         marker=dict(color='green', symbol='triangle-up', size=10),
                                         text=[f'{data["Profit/Loss"].iloc[i]:.2f}'], textposition='top center'))
            elif data['Profit/Loss'].iloc[i] < 0:
                fig.add_trace(go.Scatter(x=[data.index[i]], y=[data['Close'].iloc[i]], mode='markers+text',
                                         marker=dict(color='red', symbol='triangle-down', size=10),
                                         text=[f'{data["Profit/Loss"].iloc[i]:.2f}'], textposition='bottom center'))

        fig.update_layout(title=f'Live Prices and Profit/Loss for {ticker}', xaxis_title='Date', yaxis_title='Price',
                          xaxis_rangeslider_visible=False, template='plotly_white')

        st.plotly_chart(fig, use_container_width=True)

    plot_live_data(stock_data)

    # Function to plot with indicators
    def plot_with_indicators(data, forecast_df, title):
        fig = go.Figure()

        # Plot historical prices
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Prices', line=dict(color='blue')))

        # Plot forecasted prices
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecasted Prices', line=dict(color='orange')))
        
        if 'Lower Bound' in forecast_df.columns and 'Upper Bound' in forecast_df.columns:
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Lower Bound'], mode='lines', name='Lower Bound', line=dict(color='orange', dash='dash')))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Upper Bound'], mode='lines', name='Upper Bound', line=dict(color='orange', dash='dash')))
        else:
            st.warning("Forecast Data does not include 'Lower Bound' and 'Upper Bound' columns.")

        # Add arrows for price movement
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
                fig.add_annotation(x=data.index[i], y=data['Close'].iloc[i], ax=data.index[i - 1], ay=data['Close'].iloc[i - 1],
                                   arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='green', showarrow=True)
            elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
                fig.add_annotation(x=data.index[i], y=data['Close'].iloc[i], ax=data.index[i - 1], ay=data['Close'].iloc[i - 1],
                                   arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='red', showarrow=True)

        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price', template='plotly_white')

        st.plotly_chart(fig, use_container_width=True)

    # Forecasting models
    def forecast_with_arima(data, steps=30):
        model = ARIMA(data['Close'], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        forecast_df = pd.DataFrame({'Forecast': forecast})
        forecast_df.index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps, freq='B')
        return forecast_df

    def forecast_with_sarima(data, steps=30):
        # Adjusted seasonal_order to avoid overlap with non-seasonal order
        model = SARIMAX(data['Close'], order=(5, 1, 0), seasonal_order=(1, 1, 1, 5))
        model_fit = model.fit(disp=False)
        forecast = model_fit.get_forecast(steps=steps)
        forecast_df = forecast.summary_frame()
        forecast_df = forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']]
        forecast_df.columns = ['Forecast', 'Lower Bound', 'Upper Bound']
        forecast_df.index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps, freq='B')
        return forecast_df

    # Forecasting options
    forecast_method = st.sidebar.radio("Choose Forecasting Method", ["ARIMA", "SARIMA"])

    if st.sidebar.button("Generate Forecast"):
        if forecast_method == "ARIMA":
            forecast_df = forecast_with_arima(stock_data)
        else:
            forecast_df = forecast_with_sarima(stock_data)
        plot_with_indicators(stock_data, forecast_df, f'{ticker} - {forecast_method} Forecast')
else:
    st.write("No data available. Please select a data source or upload a CSV file.")
