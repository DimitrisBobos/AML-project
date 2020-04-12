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
import graphviz
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



#Decision tree 1
clfDT =  tree.DecisionTreeClassifier(criterion='gini', max_depth=None)

#Classifier training                 
clfDT.fit(X_train, Y_train)

#  Test the trained model on the training set
Y_train_pred_DT=clfDT.predict(X_train)

# Test the trained model on the test set
Y_test_pred_DT=clfDT.predict(X_test)


# Confusion matrixes for tree 1
confMatrixTrainDT=confusion_matrix(Y_train, Y_train_pred_DT)
confMatrixTestDT=confusion_matrix(Y_test, Y_test_pred_DT)


#Evaluation for tree 1
print ('\tClassifier Evaluation')

print ('Accuracy Train=', accuracy_score(Y_train, Y_train_pred_DT, normalize=True))
print ('Accuracy Test=', accuracy_score(Y_test, Y_test_pred_DT, normalize=True))

print ('train: Conf matrix Decision Tree')
print (confMatrixTrainDT)
print (confMatrixTestDT)

 

# Measures of performance: Precision, Recall, F1 for tree 1
print ('Tree train:  Macro Train Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_train, Y_train_pred_DT, average='macro'))
print ('Tree test:  Macro Test Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_test_pred_DT, average='macro'))
print ()

print ('train-Macro-Precision-Recall-FScore',precision_recall_fscore_support(Y_train, Y_train_pred_DT, average='macro'))
print ('test-Macro-Precision-Recall-FScore',precision_recall_fscore_support(Y_test, Y_test_pred_DT, average='macro'))
print ('\n')
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_train_pred_DT)))

pr_y_test_pred_DT=clfDT.predict_proba(X_test)
print(pr_y_test_pred_DT)
fprDT, tprDT, thresholdsDT = roc_curve(Y_test, pr_y_test_pred_DT[:,1])

tree.export_graphviz(clfDT, out_file='tree1.dot')
with open("tree1.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

g = graphviz.Source(dot_graph)
g.view()

#Decision Tree size 0.5   2




# splitting thedat sets into training and tests
X_train, X_test, Y_train, Y_test = train_test_split (X_out,x_train[targetCol], test_size=size)



#Decision tree 2
clfDT1 =  tree.DecisionTreeClassifier(criterion='gini', max_depth=3)

#Classifier training                 
clfDT1.fit(X_train, Y_train)

#  Test the trained model on the training set
Y_train_pred_DT1=clfDT1.predict(X_train)

# Test the trained model on the test set
Y_test_pred_DT1=clfDT1.predict(X_test)


# Confusion matrixes for tree 2
confMatrixTrainDT1=confusion_matrix(Y_train, Y_train_pred_DT1)
confMatrixTestDT1=confusion_matrix(Y_test, Y_test_pred_DT1)


#Evaluation for tree 2
print ('\tClassifier Evaluation')

print ('Accuracy Train=', accuracy_score(Y_train, Y_train_pred_DT1, normalize=True))
print ('Accuracy Test=', accuracy_score(Y_test, Y_test_pred_DT1, normalize=True))

print ('train: Conf matrix Decision Tree')
print (confMatrixTrainDT1)
print(confMatrixTestDT1)





# Measures of performance: Precision, Recall, F1 for tree 1
print ('Tree train:  Macro Train Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_train, Y_train_pred_DT1, average='macro'))
print ('Tree test:  Macro Test Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_test_pred_DT1, average='macro'))
print ()



pr_y_test_pred_DT1=clfDT1.predict_proba(X_test)
print(pr_y_test_pred_DT1)
fprDT1, tprDT1, thresholdsDT1 = roc_curve(Y_test, pr_y_test_pred_DT1[:,1])
tree.export_graphviz(clfDT1, out_file='tree2.dot')
with open("tree2.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

g = graphviz.Source(dot_graph)
g.view()



# Decision Tree with size 0.2 

size2=0.2
rows, cols = X_out.shape

# splitting thedat sets into training and tests
X_train, X_test, Y_train, Y_test = train_test_split (X_out,x_train[targetCol], test_size=size2)



#Decision tree 1
clfDT2 =  tree.DecisionTreeClassifier(criterion='gini', max_depth=None)

#Classifier training                 
clfDT2.fit(X_train, Y_train)

#  Test the trained model on the training set
Y_train_pred_DT2=clfDT2.predict(X_train)

# Test the trained model on the test set
Y_test_pred_DT2=clfDT2.predict(X_test)


# Confusion matrixes for tree 1
confMatrixTrainDT2=confusion_matrix(Y_train, Y_train_pred_DT2)
confMatrixTestDT2=confusion_matrix(Y_test, Y_test_pred_DT2)


#Evaluation for tree 1
print ('\tClassifier Evaluation')

print ('Accuracy Train=', accuracy_score(Y_train, Y_train_pred_DT2, normalize=True))
print ('Accuracy Test=', accuracy_score(Y_test, Y_test_pred_DT2, normalize=True))

print ('train: Conf matrix Decision Tree')
print (confMatrixTrainDT2)
print (confMatrixTestDT2)




# Measures of performance: Precision, Recall, F1 for tree 1
print ('Tree train:  Macro Train Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_train, Y_train_pred_DT2, average='macro'))
print ('Tree test:  Macro Test Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_test_pred_DT2, average='macro'))
print ()




pr_y_test_pred_DT2=clfDT2.predict_proba(X_test)
print(pr_y_test_pred_DT2)
fprDT2, tprDT2, thresholdsDT2 = roc_curve(Y_test, pr_y_test_pred_DT2[:,1])
tree.export_graphviz(clfDT2, out_file='tree3.dot')

with open("tree3.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

g = graphviz.Source(dot_graph)
g.view()



 #->ROC curve for tree 



lw=2
plt.figure(10)
plt.plot(fprDT,tprDT,color='blue')
plt.plot(fprDT1,tprDT1,color='green')
plt.plot(fprDT2,tprDT2,color='red')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve using DecisionTreeClassification with different max depths')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()




