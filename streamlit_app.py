# pip install streamlit yfinance plotly statsmodels pandas
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from plotly import graph_objs as go
import statsmodels.api as sm

START = "2018-10-09"  # Adjust the start date as per available historical data
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Predication Web App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'AMZN', 'FB', 'TSLA', 'MA', 'PYPL')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start=START, end=TODAY)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'], name="stock_close"))
    fig.layout.update(
        title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Time series forecasting using statsmodels
df_train = data[['Close']].copy()
df_train.columns = ['y']
df_train['ds'] = df_train.index

model = sm.tsa.ExponentialSmoothing(df_train['y'], trend='add')
model_fit = model.fit()

future_dates = pd.date_range(
    start=df_train['ds'].iloc[-1] + timedelta(days=1), periods=period+1, freq='B')
forecast = model_fit.forecast(period+1)

forecast_data = pd.DataFrame({'ds': future_dates, 'y': forecast})
forecast_data.set_index('ds', inplace=True)

st.write(f'Forecast data for {n_years} years')
st.write(forecast_data.tail())

# Plot forecast
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], name="Actual"))
fig1.add_trace(go.Scatter(x=forecast_data.index,
               y=forecast_data['y'], name="Forecast"))
fig1.layout.update(
    title_text=f"Forecast plot for {n_years} years", xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)

# Add the footer
st.markdown("<p style='text-align: center;'>@ My Website | Designed By Aravind</p>",
            unsafe_allow_html=True)

# Save requirements.txt file
with open('requirements.txt', 'w') as f:
    f.write('streamlit\n')
    f.write('yfinance\n')
    f.write('pandas\n')
    f.write('plotly\n')
    f.write('statsmodels\n')
