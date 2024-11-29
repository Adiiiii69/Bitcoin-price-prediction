import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import requests
import plotly.graph_objs as go

# Load the data
data = pd.read_csv('C:/Users/LENOVO/OneDrive\Desktop/Data_Sci/BTC-USD.csv', parse_dates=['Date'])
data = data.sort_values('Date')

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
data['Close_scaled'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Create the input and output sequences for the LSTM model
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data)-sequence_length-1):
        X.append(data[i:(i+sequence_length)])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

sequence_length = 60
X_train, y_train = create_sequences(train_data['Close_scaled'].values, sequence_length)
X_test, y_test = create_sequences(test_data['Close_scaled'].values, sequence_length)

# Reshape the input sequences for the LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
model.fit(X_train, y_train, epochs=1, batch_size=32) ###epoch as 1 for now.
def Bitcoin_price():

    # set the API endpoint URL and parameters
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    parameters = {
        "symbol": "BTC",
        "convert": "USD",
        "CMC_PRO_API_KEY": "2131d285-7fcd-4650-8c0c-017500f48c93"
    }

    # send the API request and store the response
    response = requests.get(url, params=parameters)

    # extract the latest data point for Bitcoin from the response
    data = response.json()
    latest_data_point = data["data"]["BTC"]["quote"]["USD"]["price"]

    print(f"Latest Bitcoin price: {latest_data_point} USD")
def get_latest_bitcoin_price():
    # set the API endpoint URL and parameters
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    parameters = {
        "symbol": "BTC",
        "convert": "USD",
        "CMC_PRO_API_KEY": "2131d285-7fcd-4650-8c0c-017500f48c93"
    }

    # send the API request and store the response
    response = requests.get(url, params=parameters)

    # extract the latest data point for Bitcoin from the response
    data = response.json()
    latest_data_point = data["data"]["BTC"]["quote"]["USD"]["price"]

    return latest_data_point


#Actual_price= latest_data_point


# Predict the price for tomorrow
def prediction():
    current_price = Bitcoin_price() # Replace with the current Bitcoin price
    last_60_days = data[-60:].copy()
    last_60_days['Close_scaled'] = scaler.transform(last_60_days['Close'].values.reshape(-1, 1))
    input_data = last_60_days['Close_scaled'].values.reshape(1, -1, 1)
    predicted_price_scaled = model.predict(input_data)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]
    print("Predicted Bitcoin price for tomorrow:", predicted_price)
    return predicted_price
predicted_price = prediction()
def only_price():
    predicted_price = prediction()
    return predicted_price
    prediction()
    bitcoin1=predicted_price
    return bitcoin1
only_price()




# Set page title
st.set_page_config(page_title='Crypto Price Prediction (BITCOIN)', page_icon=':money_with_wings:')

# Add page header
st.markdown(
    f"<h1 style='color:gold; padding:10px; text-align:center;'>Crypto Price Prediction</h1>",
    unsafe_allow_html=True,
)


# Add a sidebar to the app
sidebar = st.sidebar

# Define a list of cryptocurrencies to choose from
cryptocurrencies = ['BTC']

# Add a dropdown menu to the sidebar to select a cryptocurrency
selected_crypto = sidebar.selectbox('Select a cryptocurrency', cryptocurrencies)

# Load the price data for the selected cryptocurrency
df = pd.read_csv(f'C:/Users/LENOVO/OneDrive/Desktop/Streamlit/{selected_crypto}-USD.csv')

# Display the latest price for the selected cryptocurrency
latest_price = get_latest_bitcoin_price()
st.header(f"Latest {selected_crypto} Price: ${latest_price:,.2f}")

# Create a Plotly line chart of the price history for the selected cryptocurrency
#fig = go.Figure(data=[go.Scatter(x=df['Date'], y=df['Close'])])
#fig.update_layout(title=f'{selected_crypto} Price History', xaxis_title='Date', yaxis_title='Price ($)')
#st.plotly_chart(fig)
# Add a button that, when clicked, will display the predicted price

# Create a Plotly line chart of the price history for the selected cryptocurrency
fig = go.Figure(data=[go.Scatter(x=df['Date'], y=df['Close'], line=dict(color='#ffa500', width=3))])
fig.update_layout(
    title={
        'text': f"{selected_crypto} Price History",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Date",
    yaxis_title="Price ($)",
    hovermode="x unified",
    plot_bgcolor="#f2f2f2",
    paper_bgcolor="#f2f2f2",
    font=dict(
        family="Arial",
        size=12,
        color="#333333"
    ),
    margin=dict(
        l=50,
        r=50,
        b=50,
        t=80,
        pad=4
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="#d9d9d9",
        gridwidth=1,
        linecolor="#999999",
        linewidth=1,
        ticks="outside",
        tickcolor="#999999",
        tickwidth=1,
        ticklen=5,
        tickfont=dict(
            family="Arial",
            size=10,
            color="#333333"
        )
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#d9d9d9",
        gridwidth=1,
        linecolor="#999999",
        linewidth=1,
        ticks="outside",
        tickcolor="#999999",
        tickwidth=1,
        ticklen=5,
        tickfont=dict(
            family="Arial",
            size=10,
            color="#333333"
        ),
        range=[min(df['Close']) * 0.9, max(df['Close']) * 1.1]
    )
)
# Update chart layout
fig.update_layout(
    title=f'{selected_crypto} Price History', 
    xaxis_title='Date', 
    yaxis_title='Price ($)',
    plot_bgcolor='#1f1f1f',
    paper_bgcolor='#1f1f1f',
    font=dict(color='silver')
)

# Update axis and grid line color
fig.update_xaxes(showgrid=True, gridcolor='silver', zerolinecolor='silver')
fig.update_yaxes(showgrid=True, gridcolor='silver', zerolinecolor='silver')

st.plotly_chart(fig)




if st.button("Get Predicted Price: "):
    predicted_price1 = only_price()
    if predicted_price is not None:
        st.write(f"The predicted price of Bitcoin is {predicted_price:.2f} USD")
    else:
        st.write("Error: Failed to predict the price.")

   