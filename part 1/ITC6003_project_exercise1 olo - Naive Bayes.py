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

# loading data
data = pd.read_csv('arrhythmia.data', header = None)
print ('Percentage-missing=',pd.DataFrame({'percent_missing': data.isnull().sum() * 100 / len(data)}))
# replacing ? in the raw data with the vale np.nan to denote the missing values
data = data.replace(to_replace='?', value =np.NaN)

TrainData = data.fillna(data.median())
TrainData.drop(TrainData.columns[[13]], axis=1, inplace=True)
# checking and remove rows with missing data

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


seed = 2
np.random.seed(seed)


# checking the shape of the dataset
size=0.5
rows, cols = x_train.shape

# splitting thedat sets into training and tests
X_train, X_test, Y_train, Y_test = train_test_split (X_out,x_train[targetCol], test_size=size)

#Define a Naive Bayes1
clfNB1 = GaussianNB()

#train the classifier
clfNB1.fit(X_train, Y_train)

#test the trained model on the test set
Y_test_pred_NB1=clfNB1.predict(X_test)



confMatrixTestNB1=confusion_matrix(Y_test, Y_test_pred_NB1, labels=None)


print ('Accuracy Test=', accuracy_score(Y_test, Y_test_pred_NB1, normalize=True))


print ('Conf matrix Naive Bayes')

print (confMatrixTestNB1)

# Measures of performance: Precision, Recall, F1
print ('NB: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_test, Y_test_pred_NB1, average='macro'))
print ('NB: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_test_pred_NB1, average='micro'))
print ('\n')

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred_NB1)))

pr_y_test_pred_NB1=clfNB1.predict_proba(X_test)
print(pr_y_test_pred_NB1)
fprNB1, tprNB1, thresholdsNB1 = roc_curve(Y_test, pr_y_test_pred_NB1[:,1])


    

size1=0.3
rows, cols = x_train.shape

# splitting thedat sets into training and tests
X_train, X_test, Y_train, Y_test = train_test_split (X_out,x_train[targetCol], test_size=size1)

#Define a Naive Bayes2
clfNB2 = GaussianNB()

#train the classifier
clfNB2.fit(X_train, Y_train)

#test the trained model on the test set
Y_test_pred_NB2=clfNB2.predict(X_test)



confMatrixTestNB2=confusion_matrix(Y_test, Y_test_pred_NB2, labels=None)

print ('Accuracy Test=', accuracy_score(Y_test, Y_test_pred_NB2, normalize=True))


print ('Conf matrix Naive Bayes')

print (confMatrixTestNB2)
# Measures of performance: Precision, Recall, F1
print ('NB: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_test, Y_test_pred_NB2, average='macro'))
print ('NB: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_test_pred_NB2, average='micro'))
print ('\n')
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred_NB2)))


pr_y_test_pred_NB2=clfNB2.predict_proba(X_test)
print(pr_y_test_pred_NB2)
fprNB2, tprNB2, thresholdsNB2 = roc_curve(Y_test, pr_y_test_pred_NB2[:,1])




size2=0.2
rows, cols = x_train.shape

# splitting thedat sets into training and tests
X_train, X_test, Y_train, Y_test = train_test_split (X_out,x_train[targetCol], test_size=size2)

#Define a Naive Bayes3
clfNB3 = GaussianNB()

#train the classifier
clfNB3.fit(X_train, Y_train)

#test the trained model on the test set
Y_test_pred_NB3=clfNB3.predict(X_test)



confMatrixTestNB3=confusion_matrix(Y_test, Y_test_pred_NB3, labels=None)

print ('Accuracy Test=', accuracy_score(Y_test, Y_test_pred_NB3, normalize=True))


print ('Conf matrix Naive Bayes')

print (confMatrixTestNB3)

# Measures of performance: Precision, Recall, F1
print ('NB: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_test, Y_test_pred_NB3, average='macro'))
print ('NB: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_test_pred_NB3, average='micro'))
print ('\n')

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred_NB3)))


pr_y_test_pred_NB3=clfNB3.predict_proba(X_test)
print(pr_y_test_pred_NB3)
fprNB3, tprNB3, thresholdsNB3 = roc_curve(Y_test, pr_y_test_pred_NB3[:,1])









 #->ROC curve for NB
lw=2
plt.figure(10)
plt.plot(fprNB1,tprNB1,color='blue')
plt.plot(fprNB2,tprNB2,color='green')
plt.plot(fprNB3,tprNB3,color='red')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve using NaiveBayesClassification with different test sizes')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()