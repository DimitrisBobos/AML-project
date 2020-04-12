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

# Perform PCA
#pca = PCA(n_components=5)
#X_r = pca.fit(X_out).transform(X_out)
#print('explained variance ratio (components): %s'
  #    % str(pca.explained_variance_ratio_))
# Define the parameters of PCA

# making all columns with 0-mean, and 1-std


# checking the shape of the dataset
size=0.5
rows, cols = x_train.shape

# splitting thedat sets into training and tests
X_train, X_test, Y_train, Y_test = train_test_split (X_out,x_train[targetCol], test_size=size)

#Define Support vector machine1
clfSVM1= svm.SVC(kernel='poly', probability=True)

#train the classifiers
clfSVM1.fit(X_train, Y_train)

#test the trained model on the test set
Y_test_pred_SVM1=clfSVM1.predict(X_test)

print ('Accuracy Test=', accuracy_score(Y_test, Y_test_pred_SVM1, normalize=True))
confMatrixTrainSVM1=confusion_matrix(Y_train, Y_test_pred_SVM1)
confMatrixTestSVM1=confusion_matrix(Y_test, Y_test_pred_SVM1, labels=None)

print ('Conf matrix Support Vector Classifier')

print (confMatrixTestSVM1)


# Measures of performance: Precision, Recall, F1
print ('SVM1: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_test, Y_test_pred_SVM1, average='macro'))
print ('SVM1: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_test_pred_SVM1, average='micro'))
print ('\n')

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred_SVM1)))

pr_y_test_pred_SVM1=clfSVM1.predict_proba(X_test)
fprSVM1, tprSVM1, thresholdsSVM1 = roc_curve(Y_test, pr_y_test_pred_SVM1[:,1])





size1=0.3
rows, cols = x_train.shape

# splitting thedat sets into training and tests
X_train, X_test, Y_train, Y_test = train_test_split (X_out,x_train[targetCol], test_size=size1)

#Define Support vector machine2
clfSVM2= svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=5, gamma='auto', kernel='poly',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

#train the classifiers
clfSVM2.fit(X_train, Y_train)

#test the trained model on the test set
Y_test_pred_SVM2=clfSVM2.predict(X_test)


print ('Accuracy Test=', accuracy_score(Y_test, Y_test_pred_SVM2, normalize=True))

confMatrixTestSVM2=confusion_matrix(Y_test, Y_test_pred_SVM2, labels=None)

print ('Conf matrix Support Vector Classifier')

print (confMatrixTestSVM2)


# Measures of performance: Precision, Recall, F1
print ('SVM2: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_test, Y_test_pred_SVM2, average='macro'))
print ('SVM2: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_test_pred_SVM2, average='micro'))
print ('\n')

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred_SVM2)))


pr_y_test_pred_SVM2=clfSVM2.predict_proba(X_test)
fprSVM2, tprSVM2, thresholdsSVM2 = roc_curve(Y_test, pr_y_test_pred_SVM2[:,1])





#Define Support vector machine3

clfSVM3= svm.SVC( kernel='sigmoid', probability=True)

#train the classifiers
clfSVM3.fit(X_train, Y_train)

#test the trained model on the test set
Y_test_pred_SVM3=clfSVM3.predict(X_test)

print ('Accuracy Test=', accuracy_score(Y_test, Y_test_pred_SVM3, normalize=True))

confMatrixTestSVM3=confusion_matrix(Y_test, Y_test_pred_SVM3, labels=None)

print ('Conf matrix Support Vector Classifier')

print (confMatrixTestSVM3)


# Measures of performance: Precision, Recall, F1
print ('SVM3: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_test, Y_test_pred_SVM3, average='macro'))
print ('SVM3: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_test_pred_SVM3, average='micro'))
print ('\n')


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred_SVM3)))



pr_y_test_pred_SVM3=clfSVM3.predict_proba(X_test)
fprSVM3, tprSVM3, thresholdsSVM3 = roc_curve(Y_test, pr_y_test_pred_SVM3[:,1])




#Define Support vector machine3
clfSVM4= svm.SVC( kernel='linear', class_weight='balanced', probability=True)

#train the classifiers
clfSVM4.fit(X_train, Y_train)

#test the trained model on the test set
Y_test_pred_SVM4=clfSVM4.predict(X_test)

print ('Accuracy Test=', accuracy_score(Y_test, Y_test_pred_SVM4, normalize=True))

confMatrixTestSVM4=confusion_matrix(Y_test, Y_test_pred_SVM4, labels=None)

print ('Conf matrix Support Vector Classifier')

print (confMatrixTestSVM4)


# Measures of performance: Precision, Recall, F1
print ('SVM3: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_test, Y_test_pred_SVM4, average='macro'))
print ('SVM3: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_test_pred_SVM4, average='micro'))
print ('\n')


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred_SVM4)))



pr_y_test_pred_SVM4=clfSVM4.predict_proba(X_test)
fprSVM4, tprSVM4, thresholdsSVM4 = roc_curve(Y_test, pr_y_test_pred_SVM4[:,1])

# ROC curve plotting for SVM
lw=2
plt.figure(10)
plt.plot(fprSVM1,tprSVM1,color='blue')
plt.plot(fprSVM2,tprSVM2,color='green')
plt.plot(fprSVM3,tprSVM3,color='red')
plt.plot(fprSVM4,tprSVM4,color='yellow')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve using SVMClassification with different parameters')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()