from sklearn import metrics, datasets, tree, svm, preprocessing
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_curve, auc, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection  import VarianceThreshold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
# loading data
data = pd.read_csv('arrhythmia.data', header = None)
print ('Percentage-missing=',pd.DataFrame({'percent_missing': data.isnull().sum() * 100 / len(data)}))
# replacing ? in the raw data with the vale np.nan to denote the missing values
data = data.replace(to_replace='?', value =np.NaN)

TrainData = data.fillna(data.median())
TrainData.drop(TrainData.columns[[13]], axis=1, inplace=True)
# checking and remove rows with missing data
TrainData1 = TrainData
TrainData.reset_index()

y = TrainData.iloc[:,278]
y.replace(1,0, inplace=True)
y.replace(2,1, inplace=True)
y.replace(3,1, inplace=True)
y.replace(4,1, inplace=True)
y.replace(5,1, inplace=True)
y.replace(6,1, inplace=True)
y.replace(7,1, inplace=True)
y.replace(8,1, inplace=True)
y.replace(9,1, inplace=True)
y.replace(10,1, inplace=True)
y.replace(11,1, inplace=True)
y.replace(12,1, inplace=True)
y.replace(13,1, inplace=True)
y.replace(14,1, inplace=True)
y.replace(15,1, inplace=True)
y.replace(16,1, inplace=True)


# Percentage of variance explained for each components


print(TrainData.shape)
print(TrainData.head())
selFeatures = list(TrainData.columns.values)
print ('Features=',selFeatures)
for i in range(len(selFeatures)):
    print(i, selFeatures[i])
targetCol = selFeatures[278]
print(selFeatures[278])
del selFeatures[278]
print("Target Class: '", targetCol , "'")
print(targetCol)
print(TrainData.iloc[:,278])
x = TrainData
print(x.shape)

x1 = preprocessing.scale(x)
x1 = x.astype(int)
y1 = targetCol.astype(int)
x_train = x1
x_train.shape


selector = VarianceThreshold(threshold=2.5)
X_out=selector.fit_transform(x_train)

print(X_out)

# Perform PCA
#pca = PCA(n_components=5)
#X_r = pca.fit(X_out).transform(X_out)
#print('explained variance ratio (components): %s'
  #    % str(pca.explained_variance_ratio_))
# Define the parameters of PCA

# making all columns with 0-mean, and 1-std


# checking the shape of the dataset
#Decision Tree size 0.5   1
seed = 2
np.random.seed(seed)
size=0.5
rows, cols = X_out.shape

# splitting thedat sets into training and tests
X_train, X_test, Y_train, Y_test = train_test_split (X_out,x_train[targetCol], test_size=size)








#Random Forest 1

clfRFT1 = RandomForestClassifier(n_estimators=5, max_depth=None)
clfRFT1.fit(X_train, Y_train)
Y_train_pred_RFT1=clfRFT1.predict(X_train)



print ()

print ('Accuracy Train=', accuracy_score(Y_train, Y_train_pred_RFT1, normalize=True))
print ('Accuracy Test=', accuracy_score(Y_test, Y_train_pred_RFT1, normalize=True))
confMatrixTrainRFT1=confusion_matrix(Y_train, Y_train_pred_RFT1)
confMatrixTestRFT1=confusion_matrix(Y_test, Y_train_pred_RFT1)
print(confMatrixTrainRFT1)
print(confMatrixTestRFT1)

print ('Random Forest Train: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_train, Y_train_pred_RFT1, average='macro'))
print ('Random Forest Test: Macro Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_train_pred_RFT1, average='macro'))
print ('\n')

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_train_pred_RFT1)))

pr_y_test_pred_RFT1=clfRFT1.predict_proba(X_test)
fprRFT1, tprRFT1, thresholdsRFT1 = roc_curve(Y_test, pr_y_test_pred_RFT1[:,1])



#Random Forest 2

clfRFT2 = RandomForestClassifier(n_estimators=5, max_depth=6)
clfRFT2.fit(X_train, Y_train)
Y_train_pred_RFT2=clfRFT2.predict(X_train)

print ('Accuracy Train=', accuracy_score(Y_train, Y_train_pred_RFT2, normalize=True))
print ('Accuracy Test=', accuracy_score(Y_test, Y_train_pred_RFT2, normalize=True))
confMatrixTrainRFT2=confusion_matrix(Y_train, Y_train_pred_RFT2)
confMatrixTestRFT2=confusion_matrix(Y_test, Y_train_pred_RFT2)
print(confMatrixTrainRFT2)
print(confMatrixTestRFT2)

print ('Random Forest Train: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_train, Y_train_pred_RFT2, average='macro'))
print ('Random Forest Test: Macro Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_train_pred_RFT2, average='macro'))
print ('\n')


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_train_pred_RFT2)))

pr_y_test_pred_RFT2=clfRFT2.predict_proba(X_test)
fprRFT2, tprRFT2, thresholdsRFT2 = roc_curve(Y_test, pr_y_test_pred_RFT2[:,1])

#Random Forest 3

clfRFT3 = RandomForestClassifier(n_estimators=500, max_depth=None ,max_features='sqrt')
clfRFT3.fit(X_train, Y_train)

Y_train_pred_RFT3=clfRFT3.predict(X_train)



print ('Accuracy Train=', accuracy_score(Y_train, Y_train_pred_RFT3, normalize=True))
print ('Accuracy Test=', accuracy_score(Y_test, Y_train_pred_RFT3, normalize=True))
confMatrixTrainRFT3=confusion_matrix(Y_train, Y_train_pred_RFT3)
confMatrixTestRFT3=confusion_matrix(Y_test, Y_train_pred_RFT3)
print(confMatrixTrainRFT3)
print(confMatrixTestRFT3)

print ('Random Forest Train: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_train, Y_train_pred_RFT3, average='macro'))
print ('Random Forest Test: Macro Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_train_pred_RFT3, average='macro'))
print ('\n')
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_train_pred_RFT3)))


pr_y_test_pred_RFT3=clfRFT3.predict_proba(X_test)
fprRFT3, tprRFT3, thresholdsRFT3 = roc_curve(Y_test, pr_y_test_pred_RFT3[:,1])



# ROC curve plotting for RANDOM FOREST
lw=2
plt.figure(12)
plt.plot(fprRFT1,tprRFT1,color='blue')
plt.plot(fprRFT2,tprRFT2,color='green')
plt.plot(fprRFT3,tprRFT3,color='red')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve using RandomForestClassification with different parameters')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()




