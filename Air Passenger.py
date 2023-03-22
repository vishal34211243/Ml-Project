#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv(r"C:\Users\visha\Downloads\Air-Pax Handson-2023\Handson\AirPassengers.csv")


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data['Month'] = pd.to_datetime(data['Month'])


# In[8]:


data.info()


# In[9]:


data.head()


# In[10]:


data = data.set_index('Month')


# In[11]:


data.plot()
plt.show()


# In[12]:


mean_data = data.rolling(window=12).mean()
std_data = data.rolling(window=12).std()


# In[13]:


plt.plot(data,color='blue',label='original')
plt.plot(mean_data,color='red',label='rolling_mean')
plt.plot(std_data,color='black',label='rolling_std')
plt.legend()
plt.title('rolling mean & rolling std')
plt.show()


# In[14]:


# p,d,q - Arima
# permutation and combination happening at the back end


# In[15]:


first_log = np.log(data)


# In[16]:


first_log = first_log.dropna()


# In[19]:


mean_log = first_log.rolling(window=12).mean()
std_log = first_log.rolling(window=12).std()


# In[20]:


plt.plot(first_log,color='blue',label='first_log_data')
plt.plot(mean_log,color='red',label='mean_log')
plt.plot(std_log,color='black',label='std_log')
plt.legend()
plt.show()


# In[50]:


final_data = first_log-mean_log
final_data = final_data.dropna()
final_data.head()


# In[51]:


final_mean = final_data.rolling(window=12).mean()
final_std = final_data.rolling(window=12).std()


# In[52]:


plt.plot(final_data,color='blue',label='final_data')
plt.plot(final_mean,color='red',label='final_mean')
plt.plot(final_std,color='black',label='final_std')
plt.legend(loc='best')
plt.show()


# In[25]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[37]:


decomposition = seasonal_decompose(final_data['#Passengers'])
decomposition.plot();


# In[31]:


from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf


# In[53]:


plotting_acf =acf(final_data)
plot_acf(plotting_acf);      #---- P = 1


# In[33]:


from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf


# In[54]:


plotting_pacf = pacf(final_data)
plot_pacf(plotting_pacf,lags=10);  # ----- Q = 1


# In[55]:


final_data.shape


# In[63]:


train_data = final_data.iloc[:106]['#Passengers']
test_data = final_data.iloc[107:]['#Passengers']


# In[65]:


from statsmodels.tsa.arima_model import ARIMA


# In[68]:


model = ARIMA(train_data,order=(1,0,2))
model_fit = model.fit()


# In[69]:


from statsmodels.tsa.statespace.sarimax import SARIMAX,SARIMAXResults


# In[82]:


model = SARIMAX(train_data,order=(1,0,1),seasonal_order=(1,0,1,12))
model = model.fit()


# In[88]:


final_data['predict'] = model.predict(start=len(train_data), end = len(train_data)+len(test_data),dynamic=True)
final_data[['#Passengers','predict']].plot()


# In[137]:


forecast = model.forecast(steps=120)
plt.plot(forecast,color='green')
plt.plot(final_data,color='blue')
plt.show()


# In[138]:


forecast

