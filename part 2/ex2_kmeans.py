# +----------+
# | preamble |
# +----------+

# import libraries
import numpy as np
import pandas as pd
import seaborn as sn
from IPython.display import display
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


# load the wholesale customers dataset and drop channel and region
try:
    data = pd.read_csv("Wholesale customers data.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")


# display a description of the dataset
display(data.describe())

# boxplot dataset
plt.figure(1)
data.boxplot()
plt.show()


# produce a correlation matrix for each pair of features in the data
corrMatrix = data.corr()
plt.figure(2)
sn.heatmap(corrMatrix, annot=True, linewidths=.3, cbar=True)


# produce a scatter matrix for each pair of features in the data
plt.figure(3)
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


###############################################################################

# +---------------+
# | preprocessing |
# +---------------+

# scale the data using the natural logarithm
log_data = np.log(data.copy())


# boxplot dataset after scaling
plt.figure(4)
log_data.boxplot()
plt.show()


# investigate variance of components to perform pca
pca = PCA(n_components = 6, svd_solver = 'full')
df = pca.fit_transform(log_data)


# plot explained variance per number of components
plt.figure(5)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
print('Explained variance =', pca.explained_variance_ratio_)
plt.show()


# about 83% variance is achieved for 3 components
# perform pca with 3 principal components
pca = PCA(n_components = 3, svd_solver = 'full')
X = pca.fit_transform(log_data)


# convert df from numpy array to pandas dataframe
df = pd.DataFrame(data = X,
                  columns = ['Dimension 1', 'Dimension 2', 'Dimension 3'])


# boxplot dataset after pca
plt.figure(6)
df.boxplot()
plt.show()


# remove outliers based on z-score
z = np.abs(stats.zscore(X))
df = df[(z < 3).all(axis = 1)]


# boxplot final dataset
plt.figure(7)
df.boxplot()
plt.show()


# set seed to replicate results
#seed = 1
#np.random.seed(seed)


###############################################################################

# +--------+
# | kmeans |
# +--------+

# run clusterings for different values of k to find the optimal
inertiasAll=[]
silhouettesAll=[]
for n in range(2,15):
    print ('Clustering for n=',n)
    kmeans = KMeans(n_clusters=n)
    # perform clustering
    kmeans.fit(df)
    # find the clusters the points df belong to
    y_kmeans = kmeans.predict(df)
    # get cluster centers
    centers = kmeans.cluster_centers_
    # evalute
    print ('inertia=',kmeans.inertia_)
    silhouette_values = silhouette_samples(df, y_kmeans)
    print ('silhouette=', np.mean(silhouette_values))    
    inertiasAll.append(kmeans.inertia_)
    silhouettesAll.append(np.mean(silhouette_values))


# plot silhouettes
plt.figure(8)
plt.plot(range(2,15),silhouettesAll,'r*-')
plt.ylabel('Silhouette score')
plt.xlabel('Number of clusters')
plt.show()


#plot inertias
plt.figure(9)
plt.plot(range(2,15),inertiasAll,'g*-')
plt.ylabel('Inertia Score')
plt.xlabel('Number of clusters')
plt.show()


# optimal combination of low inertia and high silhouette achieved for 9 clusters
kmeans = KMeans(n_clusters=9)

# perform kmeans clustering
kmeans.fit(df)


# find the clusters the points df belong to
y_kmeans = kmeans.predict(df)



# get cluster centers
centers = kmeans.cluster_centers_


# add cluster attribute to df dataset
df['kmeans_cluster'] = pd.Series(y_kmeans, index=df.index)



# plot data accoding to the cluster they belong
# -> cluster_0
df_c0 = df.where(df['kmeans_cluster'] == 0)
df_c0 = df_c0.drop(columns = 'kmeans_cluster')
df_c0 = df_c0.dropna()
plt.figure(10)
df_c0.boxplot()
plt.title('Cluster_0 visualization')
plt.show()


# -> cluster_1
df_c1 = df.where(df['kmeans_cluster'] == 1)
df_c1 = df_c1.drop(columns = 'kmeans_cluster')
df_c1 = df_c1.dropna()
plt.figure(11)
df_c1.boxplot()
plt.title('Cluster_1 visualization')
plt.show()


# -> cluster_2
df_c2 = df.where(df['kmeans_cluster'] == 2)
df_c2 = df_c2.drop(columns = 'kmeans_cluster')
df_c2 = df_c2.dropna()
plt.figure(12)
df_c2.boxplot()
plt.title('Cluster_2 visualization')
plt.show()


# -> cluster_3
df_c3 = df.where(df['kmeans_cluster'] == 3)
df_c3 = df_c3.drop(columns = 'kmeans_cluster')
df_c3 = df_c3.dropna()
plt.figure(13)
df_c3.boxplot()
plt.title('Cluster_3 visualization')
plt.show()


# -> cluster_4
df_c4 = df.where(df['kmeans_cluster'] == 4)
df_c4 = df_c4.drop(columns = 'kmeans_cluster')
df_c4 = df_c4.dropna()
plt.figure(14)
df_c4.boxplot()
plt.title('Cluster_4 visualization')
plt.show()


# -> cluster_5
df_c5 = df.where(df['kmeans_cluster'] == 5)
df_c5 = df_c5.drop(columns = 'kmeans_cluster')
df_c5 = df_c5.dropna()
plt.figure(15)
df_c5.boxplot()
plt.title('Cluster_5 visualization')
plt.show()


# -> cluster_6
df_c6 = df.where(df['kmeans_cluster'] == 6)
df_c6 = df_c6.drop(columns = 'kmeans_cluster')
df_c6 = df_c6.dropna()
plt.figure(16)
df_c6.boxplot()
plt.title('Cluster_6 visualization')
plt.show()


# -> cluster_7
df_c7 = df.where(df['kmeans_cluster'] == 7)
df_c7 = df_c7.drop(columns = 'kmeans_cluster')
df_c7 = df_c7.dropna()
plt.figure(17)
df_c7.boxplot()
plt.title('Cluster_7 visualization')
plt.show()


# -> cluster_8
df_c8 = df.where(df['kmeans_cluster'] == 8)
df_c8 = df_c8.drop(columns = 'kmeans_cluster')
df_c8 = df_c8.dropna()
plt.figure(18)
df_c8.boxplot()
plt.title('Cluster_8 visualization')
plt.show()


###############################################################################

