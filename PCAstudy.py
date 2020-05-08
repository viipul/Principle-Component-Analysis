import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
cancer_data=load_breast_cancer()
print(cancer_data)
print(cancer_data.keys())
print(cancer_data['DESCR'])
df=pd.DataFrame(cancer_data['data'],columns=cancer_data['feature_names'])
print(df)
from sklearn.preprocessing import StandardScaler
# The idea behind StandardScaler is that it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1.
# In case of multivariate data, this is done feature-wise (in other words independently for each column of the data).
# Given the distribution of the data, each value in the dataset will have the mean value subtracted, and then divided by the standard deviation of the whole dataset (or feature in the multivariate case).
scalar=StandardScaler()
a=scalar.fit(df)
scaled_df=scalar.transform(df)
# transform() : parameters generated from fit()\
# method,applied upon model to generate transformed data set.
print(scaled_df)
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(scaled_df)
x_pca=pca.transform(scaled_df)
print('Initial Shape',scaled_df.shape)
print('Shape ater applying PCA',x_pca.shape)
print(x_pca)

plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer_data['target'])
plt.xlabel('First principle component')
plt.ylabel('Second principle component')
plt.show()
