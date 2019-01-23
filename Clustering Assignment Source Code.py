#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random as rd
get_ipython().run_line_magic('matplotlib', 'inline')

##### Doing this in Python was too slow. Used R instead 
## Use when doing full dataset analysis
#df = pd.read_csv('Medicare_Provider_Util_Payment_PUF_CY2016.txt',sep='\t',engine='python')
## Trim Dataset by dropping unnecessary columns and filter to business relevant data only
#df.drop(['NPI','NPPES_ENTITY_CODE','NPPES_PROVIDER_LAST_ORG_NAME','NPPES_PROVIDER_FIRST_NAME','NPPES_PROVIDER_MI','NPPES_PROVIDER_STREET1','NPPES_PROVIDER_STREET2','HCPCS_DESCRIPTION'],axis=1,inplace = True)
#df = df[(df['MEDICARE_PARTICIPATION_INDICATOR']=='Y') & (df['NPPES_PROVIDER_COUNTRY']=='US') & (df['PROVIDER_TYPE']=='Diagnostic Radiology')]
#df.drop(['MEDICARE_PARTICIPATION_INDICATOR','NPPES_PROVIDER_COUNTRY','PROVIDER_TYPE'],axis=1,inplace = True)

df = pd.read_csv('MedicareData.csv')
df.drop(['Unnamed: 0'],inplace = True, axis = 1)

# Further subsegment the data to only include data in California
df = df[(df['NPPES_PROVIDER_STATE']=='CA') & (df['HCPCS_DRUG_INDICATOR']=='N')]

df.info()

df.head()

df.describe()

sns.pairplot(df)

sns.heatmap(df.corr(),cmap = 'magma',linecolor = 'white',lw=1)

from sklearn.preprocessing import OneHotEncoder
df_analyze = pd.DataFrame()
Gender_OHE = OneHotEncoder(categories = 'auto')
X = Gender_OHE.fit_transform(df.NPPES_PROVIDER_GENDER.values.reshape(-1,1)).toarray()
dfOneHot = pd.DataFrame(X,columns = ['Gender_'+str(int(i))for i in range(X.shape[1])])
dfOneHot.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)
df_analyze = pd.concat([df,dfOneHot],axis = 1)

POS_OHE = OneHotEncoder(categories = 'auto')
X = POS_OHE.fit_transform(df.PLACE_OF_SERVICE.values.reshape(-1,1)).toarray()
dfOneHot = pd.DataFrame(X,columns = ['PLACEOFSERVICE'+str(int(i))for i in range(X.shape[1])])
dfOneHot.reset_index(drop=True, inplace=True)
df_analyze = pd.concat([df_analyze,dfOneHot],axis = 1)

# drop unwanted columns from analysis
df_analyze.drop(['NPPES_CREDENTIALS','NPPES_PROVIDER_GENDER','NPPES_PROVIDER_CITY','NPPES_PROVIDER_STATE','NPPES_PROVIDER_ZIP',
                'PROVIDER_TYPE','HCPCS_CODE','HCPCS_DRUG_INDICATOR','PLACE_OF_SERVICE',
                'BENE_UNIQUE_CNT','AVERAGE_MEDICARE_ALLOWED_AMT',
                 'AVERAGE_MEDICARE_PAYMENT_AMT'], axis=1,inplace = True)

# Preprocess data using normalize function
from sklearn import preprocessing
df_normalized = preprocessing.normalize(df_analyze)

# Compute Scree plot for different number of clusters
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df_normalized)
    kmeanModel.fit(df_normalized)
    distortions.append(sum(np.min(cdist(df_normalized, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df_normalized.shape[0])
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Compute Silhoette Graph for different number of clusters to select
# optimal number of clusters
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
for n_clusters in range(2, 9):
    model = SilhouetteVisualizer(KMeans(n_clusters))
    model.fit(df_normalized)
    model.poof()

# Utlize TSNE to visualize data
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pylab as pl
num_of_clusters = 4
kmeans = KMeans(n_clusters = num_of_clusters)
kmeans.fit(df_normalized)

X = TSNE(n_components = 2).fit_transform(df_normalized)

for i in range(0, X.shape[0]):
    if kmeans.labels_[i] == 0:
        c1 = pl.scatter(X[i,0], X[i, 1], c='red')
    elif kmeans.labels_[i] == 1:
        c2 = pl.scatter(X[i,0], X[i, 1], c='green')
    elif kmeans.labels_[i] == 2:
        c3 = pl.scatter(X[i,0], X[i, 1], c='blue')
    elif kmeans.labels_[i] == 3:
        c4= pl.scatter(X[i,0], X[i, 1], c='yellow')    

pl.legend([c1, c2, c3, c4], ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'])
pl.title('K-means clusters the Medicare dataset into 4 clusters')
pl.show()

# Compute Summary Statistics
labels = pd.DataFrame(kmeans.labels_,columns = ['labels'])
df_describedata = pd.concat([df_analyze,labels],axis=1)

print(df_describedata[df_describedata['labels']==0.0].describe())
print(df_describedata[df_describedata['labels']==1.0].describe())
print(df_describedata[df_describedata['labels']==2.0].describe())
print(df_describedata[df_describedata['labels']==3.0].describe())
