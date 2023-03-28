#!/usr/bin/env python
# coding: utf-8

# In[261]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
warnings.filterwarnings('ignore')


# In[262]:


# !pip install missingno


# In[263]:


data = pd.read_csv(r"C:\Users\visha\Downloads\OnlineRetail (3).csv",encoding='unicode_escape')


# In[264]:


data.head()


# In[265]:


data.shape


# In[266]:


msno.bar(data)


# In[267]:


data.isna().sum()


# 1.) Missing values in Decscription column - 1454
# 
# 2.) Missing values in CusstomerID column - 135080

# In[268]:


data['CustomerID'].nunique()


# There are 4372  unique customers who does purchasing via online.
# 

# In[269]:


data['Country'].value_counts().reset_index()


# 1.) top 3 max orders come from Unite Kingdom, Germany,France
# 
# 2.) minimum order come from Saudi Arabia

# In[270]:


data['CustomerID'].value_counts()


# In[271]:


data['InvoiceDate'].max()


# In[272]:


data['InvoiceDate'].min()


# In[273]:


data.info()


# In[274]:


data.describe().T 


# 1.) There are Missing values in the CustomerID column
# 
# 2.) Quantity and UnitPrice minimum value suppose to be in postive, here it is in neegative, so we need to remove those negative values.

# In[275]:


data.describe(exclude=['int64','float64']).T


# The Country United Kingdom's distribution is 495478 out of 541909, the maximum bussiness is coming from United Kingdom.

# In[276]:


data.dropna(inplace=True)


# In[277]:


data.isna().sum()


# In[278]:


data.drop(data[data['Quantity']<0].index,inplace=True)


# In[279]:


data.drop(data[data['UnitPrice']<0].index,inplace= True)


# In[280]:


data.shape


# In[281]:


data[data['Quantity']<0].shape[0], data[data['UnitPrice']<0].shape[0]


# ## RFM Analysis

# In[282]:


#Total Amount spent by customer

data['Total_Amount'] = data['Quantity']*data['UnitPrice']

ta = data.groupby(by='CustomerID')['Total_Amount'].sum().reset_index()
ta


# In[283]:


# count of transaction made by each customers.

io = data.groupby(by='CustomerID')['InvoiceNo'].count().reset_index()
io


# In[284]:


# Last Transaction

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['Last_Transaction'] = (data['InvoiceDate'].max()-data['InvoiceDate']).dt.days
lt = data.groupby(['CustomerID','Country'])['Last_Transaction'].max().reset_index()
lt


# Creating RFM Dataset

# In[285]:


merge1 = pd.merge(ta,io,how='inner',on='CustomerID')
new_df = pd.merge(merge1,lt,how='inner',on='CustomerID')
new_df


# In[286]:


plt.figure(figsize=(7,5))
sns.boxplot(new_df[['Total_Amount','InvoiceNo','Last_Transaction']])
plt.show()


# In[287]:


# remove the outliers

q1 = np.percentile(new_df['Total_Amount'],25)
q3 = np.percentile(new_df['Total_Amount'],75)
IQR = q3-q1
Lower_limit = q1-1.5*IQR
upper_limit = q3+1.5*IQR

iqr_df = new_df[(new_df['Total_Amount']>=Lower_limit) & (new_df['Total_Amount']<=upper_limit)]


# In[288]:


plt.boxplot(iqr_df['Total_Amount'])
plt.show()


# In[289]:


iqr_df.reset_index(drop=True,inplace=True)


# In[290]:


new1_df=iqr_df.copy()
df_num=['Total_Amount','InvoiceNo','Last_Transaction']
for i in df_num:
    sns.displot(new1_df[i],bins=10,kde=True)
    plt.gcf().set_size_inches(15, 7)
    plt.show()


# 1.) Total Amount (Monetary) and InvoiceNo (Frequency) histogram are right-skewed.
# 
# 2.) Last_Transaction (Recency) histogram is bimodal

# In[291]:


plt.figure(figsize=(12,10))
new1_df.groupby('Country')['Total_Amount'].max().sort_values(ascending=True).plot(kind = 'barh')


# In[292]:


plt.figure(figsize=(12,10))
new1_df.groupby('Country')['Total_Amount'].min().sort_values(ascending=False).plot(kind = 'barh')


# 1.) Unitied Kingdom Spends the maximum Amount.
# 
# 2.) Lebanon spends the very minimum amount.

# In[293]:


plt.figure(figsize=(15,12))
new1_df.groupby("Country")['InvoiceNo'].mean().sort_values(ascending=True).plot(kind='barh')
plt.show()


# 1.) On Average, customers in belgium shop most frequently.
# 
# 2.) On Average, Bahrain's customers are the least frequent buyers.

# In[294]:


plt.figure(figsize=(15,20))
new1_df.groupby("Country")['Last_Transaction'].mean().sort_values(ascending=True).plot(kind='barh')
plt.show()


# 1.) On Average, Customers in  Lithuania shoped most recently.

# In[295]:


plt.figure(figsize=(10,7))
sns.heatmap(new1_df.corr(),cmap='Pastel1',annot = True)


# 1.) As the heatmap shows, Total Amount and InvoiceNo is correlated positively.

# In[296]:


plt.figure(figsize=(15,8))
sns.pairplot(new1_df, corner=True)
plt.show()


# In[297]:


from sklearn.preprocessing import MinMaxScaler


# In[298]:


new2_df = new1_df[['Total_Amount','InvoiceNo','Last_Transaction']]
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(new2_df)
scaled_df = pd.DataFrame(scaled_df)
scaled_df.columns=['Total_Amount','InvoiceNo','Last_Transaction']
scaled_df['Country'] = new1_df['Country']
scaled_df


# In[299]:


# pip install yellowbrick


# In[300]:


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


# In[301]:


df_k = scaled_df.drop(columns='Country',axis=1)

model = KMeans()

visualizer = KElbowVisualizer(model,k=(1,10) , timings=False)
visualizer.fit(df_k)
visualizer.show()


# Therefore clusters for our data is 3

# In[302]:


model = KMeans()

visualizer = KElbowVisualizer(model , k=(2,20),metric='silhouette', timings=False)
visualizer.fit(df_k)
visualizer.show()


# Therefore, by silhouette number of clusters is 3.

# In[303]:


km = KMeans(n_clusters=3)
y_pred = km.fit_predict(df_k)
df_k['Cluster'] = y_pred
df_k


# In[304]:


km.cluster_centers_


# Given above are the clusters centroids

# In[305]:


new2_df['Clusters'] = y_pred
df1 = new2_df[new2_df['Clusters']==0]
df2 = new2_df[new2_df['Clusters']==1]
df3 = new2_df[new2_df['Clusters']==2]


# In[306]:


# visualizing the data with original data.

plt.figure(figsize=(20,10))
plt.scatter(df1['Last_Transaction'],df1['Total_Amount'],color='green')
plt.scatter(df2['Last_Transaction'],df2['Total_Amount'],color='red')
plt.scatter(df3['Last_Transaction'],df3['Total_Amount'],color='blue')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()
plt.xlabel('Last_Transaction')
plt.ylabel('Total_Amount')
plt.title('K-Means Cluster Profiles,Last_transaction vs. TotalAmount')
plt.show()


# In[316]:


# visualizing the data with scaled data.

df1 = df_k[df_k['Cluster']==0]
df2 = df_k[df_k['Cluster']==1]
df3 = df_k[df_k['Cluster']==2]
plt.figure(figsize=(20,10))
plt.scatter(df1['Last_Transaction'],df1['Total_Amount'],color='green')
plt.scatter(df2['Last_Transaction'],df2['Total_Amount'],color='red')
plt.scatter(df3['Last_Transaction'],df3['Total_Amount'],color='blue')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()
plt.xlabel('Last_transaction')
plt.ylabel('TotalAmount')
plt.title('K-Means Cluster Profiles,Last_transaction vs. TotalAmount')
plt.show()


# 1.) The K-Means model segments the data into distinct clusters based on customer's Recency(Last_Transaction) and Monetary(Total_Amount).
# 
# 2.) Cluster 0 consists of customers with the last_transaction between 186 and 373 days, and total amount spent between  $3.75 and 1542.
# 
# 3.) Cluster 1 consist of customers with the last_transaction between 0 and 186 days, and total amount spent between $0 and 3192.
# 
# 4.) Cluster 2 consits of custoemrs with the last_transaction between 43 and 373 days, and total amount spent between  $1236 and 3712.

# ### Hierarchical Agglomerative Clustering

# In[309]:


import scipy.cluster.hierarchy as sch
dendo = sch.dendrogram(sch.linkage(df_k,method='ward'))


# In[333]:


from sklearn.cluster import AgglomerativeClustering
new_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
two_clusters = new_cluster.fit_predict(df_k)

new_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
three_clusters = new_cluster.fit_predict(df_k)


# In[334]:


df_k['two_clusters'] = two_clusters
df_k['three_clusters'] = three_clusters
df_k


# In[337]:


# clusters with scaled data:

df1 = df_k[df_k['two_clusters']==0]
df2 = df_k[df_k['two_clusters']==1]
plt.figure(figsize=(20,10))
plt.scatter(df1['Last_Transaction'],df1['Total_Amount'],color='green')
plt.scatter(df2['Last_Transaction'],df2['Total_Amount'],color='red')
plt.legend()
plt.xlabel('Last_Transaction')
plt.ylabel('Total_Amount')
plt.title('K-Means Cluster Profiles,Last_Transaction vs. Total_Amount')
plt.show()


# In[343]:


# clusters with scaled data:
df1 = df_k[df_k['three_clusters']==0]
df2 = df_k[df_k['three_clusters']==1]
df3 = df_k[df_k['three_clusters']==2]
plt.figure(figsize=(20,10))
plt.scatter(df1['Last_Transaction'],df1['Total_Amount'],color='green')
plt.scatter(df2['Last_Transaction'],df2['Total_Amount'],color='red')
plt.scatter(df3['Last_Transaction'],df3['Total_Amount'],color='blue')
plt.legend()
plt.xlabel('Last_transaction')
plt.ylabel('TotalAmount')
plt.title('K-Means Cluster Profiles,Last_transaction vs. TotalAmount')
plt.show()


# In[400]:


data['InvoiceDate'].sort_index()


# In[402]:


data['InvoiceDate'].max()

