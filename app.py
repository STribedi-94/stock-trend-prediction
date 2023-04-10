import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

import datetime as dt
import yfinance as yf


# Define a start date and End Date
start = dt.datetime(2011,1,1)
end =  dt.datetime(2023,3,1)

st.title('Stock Trend Prediction')

stock = st.text_input('Enter Stock Ticker','POWERGRID.NS')
 
df = yf.download(stock, start , end)

# Describing Data
st.subheader('Data from Jan 2011 to Mar 2023')
st.write(df.describe())

#Visualizations

ema20 = df.Close.ewm(span=20, adjust=False).mean()
ema50 = df.Close.ewm(span=50, adjust=False).mean()

st.subheader('Closing Price vs Time Chart with 20 & 50 Days of Exponential Moving Avarage')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close,'y')
plt.plot(ema20, 'g', label= 'EMA of 20 Days')
plt.plot(ema50, 'r', label= 'EMA of 50 Days')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


ema100 = df.Close.ewm(span=100, adjust=False).mean()
ema200 = df.Close.ewm(span=200, adjust=False).mean()

st.subheader('Closing Price vs Time Chart with 100 & 200 EMA')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ema100, 'g', label= 'EMA of 100 Days')
plt.plot(ema200, 'r', label= 'EMA of 200 Days')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


# Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)


# lOading the model

model = load_model('stock_dl_model.h5')

# Model Testing 

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test=[]
y_test=[]

for i in range (100,input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])
    
x_test,y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final PLot

st.subheader('Prediction Vs Original Trend')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, 'g', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

