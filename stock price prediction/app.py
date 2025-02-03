import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

model = load_model(r'C:\Users\muham\OneDrive\Pictures\Desktop\hamthan\data science projects\stock price prediction\stock price prediction.keras')


st.header('stock market predictor')

stock = st.text_input('Enter Stock Symbol','GOOGL')
start = '2012-01-01'
end = '2023-01-01'

data = yf.download(stock, start, end)

st.subheader('stock data')
st.write(data)


data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)]) #pd.dataframe is used to convert into 2-dimension
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

scaler = MinMaxScaler(feature_range=(0,1))


pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days,data_test],ignore_index = True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10,8))
plt.plot(ma_50_days,'r')
plt.plot(data.Close,'g')
plt.xlabel('time')
plt.ylabel('price')
st.pyplot(fig1)


x=[] #it add the input sequences=
y=[] #it add the value that to be predicted (target value)
for i in range(100,data_test_scale.shape[0]):
   x.append(data_test_scale[i-100:i]) # ex:[0:100], it add the input sequences
   y.append(data_test_scale[i,0])#row,column index, it add the value that to be predicted

x,y = np.array(x),np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict*scale #predicted price
y = y*scale #actual price

st.subheader('Original price vs predicted price')
fig2 = plt.figure(figsize=(10,8))
plt.plot(predict,'r')
plt.plot(y,'g')
plt.xlabel('time')
plt.ylabel('price')
st.pyplot(fig2)
