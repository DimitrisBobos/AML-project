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
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score


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


# boxplot dataset after pca
#plt.figure(6)
#X.boxplot()


# convert df from numpy array to pandas dataframe
df = pd.DataFrame(data = X,
                 columns = ['Dimension 1', 'Dimension 2', 'Dimension 3'])


# remove outliers based on z-score
z = np.abs(stats.zscore(X))
df = df[(z < 3).all(axis = 1)]


# boxplot final dataset
plt.figure(7)
boxplot = df.boxplot()
plt.show()


# set seed to replicate results
seed = 1
np.random.seed(seed)


###############################################################################

# +----------+
# | gaussian |
# +----------+
###calculating silhouette score
def getScore(nClusters):
    # apply gaussian mixtures clustering algorithm to df 
    clusterer = GMM(n_components=nClusters, covariance_type='full', random_state=35).fit(df)

    # predict the cluster for each data point
    preds = clusterer.predict(df)

    # find the cluster centers
    centers = clusterer.means_
    
    
    # calculate the log-likelihood for the number of clusters chosen
    score = clusterer.score(df)
    return score


scores = []
nClusts = []
for i in range(2,20):
    nClusts.append(i)
    scores.append(getScore(i))
print(scores)
probabilities = np.exp(scores)
X = pd.DataFrame({'n_clustsers':nClusts,'scores':probabilities})
X = X.set_index('n_clustsers')

print(X)
X.plot()


###calculating BIC and AIC
n_components = np.arange(2, 20)
models = [GMM(n, covariance_type='full', random_state=35).fit(df)
          for n in n_components]
plt.figure(8)
plt.plot(n_components, [m.bic(df) for m in models], label='BIC')
plt.plot(n_components, [m.aic(df) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')


clusterer = GMM(n_components=5, covariance_type='full', random_state=35).fit(df)

# predict the cluster for each data point
preds = clusterer.predict(df)

# find the cluster centers
centers = clusterer.means_

# calculate the log-likelihood for the number of clusters chosen
score = clusterer.score(df)
print(score)

#probability for every point to belong to a distribution
probs=clusterer.predict_proba(df)
print(probs)


# add cluster attribute to df dataset
df['cluster'] = pd.Series(preds, index=df.index)


# plot data accoding to the cluster they belong
# -> cluster_0
df_c0 = df.where(df['cluster'] == 0)
df_c0 = df_c0.drop(columns = 'cluster')
df_c0 = df_c0.dropna()
plt.figure(0)
boxplot = df_c0.boxplot()
plt.title('Cluster_0 visualization')
plt.show()

# -> cluster_1
df_c1 = df.where(df['cluster'] == 1)
df_c1 = df_c1.drop(columns = 'cluster')
df_c1 = df_c1.dropna()
plt.figure(1)
boxplot = df_c1.boxplot()
plt.title('Cluster_1 visualization')
plt.show()

# -> cluster_2
df_c2 = df.where(df['cluster'] == 2)
df_c2 = df_c2.drop(columns = 'cluster')
df_c2 = df_c2.dropna()
plt.figure(2)
boxplot = df_c2.boxplot()
plt.title('Cluster_2 visualization')
plt.show()

# -> cluster_3
df_c3 = df.where(df['cluster'] == 3)
df_c3 = df_c3.drop(columns = 'cluster')
df_c3 = df_c3.dropna()
plt.figure(3)
boxplot = df_c3.boxplot()
plt.title('Cluster_3 visualization')
plt.show()




###############################################################################