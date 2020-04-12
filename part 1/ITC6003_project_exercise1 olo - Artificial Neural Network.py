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


seed = 2
np.random.seed(seed)
size=0.5
rows, cols = X_out.shape

# splitting thedat sets into training and tests
X_train, X_test, Y_train, Y_test = train_test_split (X_out,x_train[targetCol], test_size=size)




# defining neural network 1
clfANN1 = MLPClassifier(solver='adam', activation='relu',
                    batch_size=1, tol=1e-19,
                    hidden_layer_sizes=(25), random_state=1, max_iter=1000, verbose=False)
   
# training the neural network 1 classifier                 
clfANN1.fit(X_train, Y_train)                         

# testing the trained neural network 1 on the test set
Y_test_pred_ANN1=clfANN1.predict(X_test)

# Confusion matrixes for neural network
confMatrixTrainANN1=confusion_matrix(Y_train, Y_test_pred_ANN1)
confMatrixTestANN1=confusion_matrix(Y_test, Y_test_pred_ANN1)

print(confMatrixTrainANN1)
print(confMatrixTestANN1)


print ('Accuracy Train=', accuracy_score(Y_train, Y_test_pred_ANN1, normalize=True))
print ('Accuracy Test=', accuracy_score(Y_test, Y_test_pred_ANN1, normalize=True))


# measures of performance for neural network 1: Precision, Recall, F1
print ('Artificial Neural Network TRAIN: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_train, Y_test_pred_ANN1, average='macro'))
print ('Artificial Neural Network TEST: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_test_pred_ANN1, average='micro'))
print ('\n')
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred_ANN1)))
# ROC curve for artificial neural network 1
pr_y_test_pred_ANN1=clfANN1.predict_proba(X_test)
fprANN1, tprANN1, thresholdsANN1 = roc_curve(Y_test, pr_y_test_pred_ANN1[:,1])










# defining neural network 2
clfANN2 = MLPClassifier(solver='adam', activation='relu',
                    batch_size=1, tol=1e-19,
                    hidden_layer_sizes=(15,), random_state=1, max_iter=1000, verbose=False)
   
# training the neural network classifier                 
clfANN2.fit(X_train, Y_train)                         

# testing the trained neural network on the test set
Y_test_pred_ANN2=clfANN2.predict(X_test)

# Confusion matrixes for neural network
confMatrixTrainANN2=confusion_matrix(Y_train, Y_test_pred_ANN2)
confMatrixTestANN2=confusion_matrix(Y_test, Y_test_pred_ANN2)

print(confMatrixTrainANN2)
print(confMatrixTestANN2)

print ('Accuracy Train=', accuracy_score(Y_train, Y_test_pred_ANN2, normalize=True))
print ('Accuracy Test=', accuracy_score(Y_test, Y_test_pred_ANN2, normalize=True))


# measures of performance for neural network 1: Precision, Recall, F1
print ('Artificial Neural Network TRAIN: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_train, Y_test_pred_ANN2, average='macro'))
print ('Artificial Neural Network TEST: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_test_pred_ANN2, average='micro'))
print ('\n')
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred_ANN2)))


# ROC curve for artificial neural network
pr_y_test_pred_ANN2=clfANN2.predict_proba(X_test)
fprANN2, tprANN2, thresholdsANN2 = roc_curve(Y_test, pr_y_test_pred_ANN2[:,1])





# defining neural network 3
clfANN3 = MLPClassifier(solver='adam', activation='relu',
                    batch_size=1, tol=1e-19,
                    hidden_layer_sizes=(35,), random_state=1, max_iter=1000, verbose=False)
   
# training the neural network classifier                 
clfANN3.fit(X_train, Y_train)                         

# testing the trained neural network on the test set
Y_test_pred_ANN3=clfANN3.predict(X_test)

# Confusion matrixes for neural network
confMatrixTrainANN3=confusion_matrix(Y_train, Y_test_pred_ANN3)
confMatrixTestANN3=confusion_matrix(Y_test, Y_test_pred_ANN3)

print(confMatrixTrainANN3)
print(confMatrixTestANN3)


print ('Accuracy Train=', accuracy_score(Y_train, Y_test_pred_ANN3, normalize=True))
print ('Accuracy Test=', accuracy_score(Y_test, Y_test_pred_ANN3, normalize=True))

# measures of performance for neural network: Precision, Recall, F1
print ('Artificial Neural Network: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_train, Y_test_pred_ANN3, average='macro'))
print ('Artificial Neural Network: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_test_pred_ANN3, average='micro'))
print ('\n')

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred_ANN3)))

# ROC curve for artificial neural network
pr_y_test_pred_ANN3=clfANN3.predict_proba(X_test)
fprANN3, tprANN3, thresholdsANN3 = roc_curve(Y_test, pr_y_test_pred_ANN3[:,1])


#

# ROC curve plotting for neural network classifiers
lw=2
plt.figure(11)
plt.plot(fprANN1,tprANN1,color='blue')
plt.plot(fprANN2,tprANN2,color='green')
plt.plot(fprANN3,tprANN3,color='red')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve using ArtificialNeuralNetworkClassification with different parameters')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()













