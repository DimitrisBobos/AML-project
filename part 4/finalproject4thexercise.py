from sklearn import metrics
from sklearn import datasets 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn import tree
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

buys = pd.read_csv('yoochoose-buys.dat', names=['SessionID', 'Timestamp', 'ItemID', 'Price', 'Quantity'])
                    
clicks = pd.read_csv('yoochoose-clicks.dat', names=['SessionID', 'Timestamp', 'ItemID', 'Category'])      

buys = buys.sort_values(by=['Timestamp', 'SessionID'])
clicks = clicks.sort_values(by=['Timestamp', 'SessionID'])

buys['Timestamp'] = pd.to_datetime(buys['Timestamp'])


clicks['Timestamp'] = pd.to_datetime(clicks['Timestamp'])




clicks = pd.concat([clicks.iloc[:100000],clicks.iloc[-100000:]])
print(len(clicks))

buys1 = buys.groupby(['SessionID','ItemID']).agg({'Price':np.mean,'Quantity':np.sum}).reset_index()
buys1.head()

clicks['prev_time'] = clicks.groupby('SessionID')['Timestamp'].transform(lambda x: x.shift())
clicks['difference'] = clicks['Timestamp'] - clicks['prev_time'] # in minutes
clicks['total_time'] = clicks.groupby('SessionID')['difference'].transform(lambda x: x.shift(-1)).dt.seconds/60
del clicks['prev_time']
del clicks['difference']


df = pd.merge(clicks,buys1,how='left',left_on=['SessionID','ItemID'],right_on=['SessionID','ItemID'])
df.shape




selFeatures = list(df.columns.values)
print ('Features=',selFeatures)





def get_category(category):
    category = str(category)
    if category == '0':
        return '1'
    elif category == "S":
        return '1'
    elif int(category) > 0 :
        return 1
    return "Error"

df['Category1'] = df['Category'].apply(lambda x: get_category(x))
del df['Category']

selFeatures = list(df.columns.values)
print ('Features=',selFeatures)



group1 = (df.groupby('SessionID').agg({'Category1': np.size, 
                              'ItemID': pd.Series.nunique,
                              'Quantity': lambda x: int(all(x.notnull()))})
                                .rename(columns={'Category1':'Total_clicks','ItemID':'unique_items_seen','Quantity':'buy'}))

group2 = (df.groupby('SessionID')
            .agg({'Category1': pd.Series.nunique,'total_time':np.min})
            .rename(columns={'Category1':'unique_categories_seen','total_time':'duration'})
         )
           
features = pd.merge(group1,group2,left_on='SessionID',right_index=True)
selFeatures = list(features.columns.values)
print ('Features=',selFeatures)
print (features.isnull().sum())
features = features.dropna()

np.random.seed(4)

y = features['buy']
del features['buy']




size=0.2
rows, cols = features.shape


#split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(features, y, test_size=size)


#Decision tree 1
clfDT =  tree.DecisionTreeClassifier(criterion='gini', max_depth=5, class_weight ='balanced')

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
print ()

print ('test: Conf matrix Nearest Neighbor')
print (confMatrixTestDT)
print ()


# Measures of performance: Precision, Recall, F1 for tree 1
print ('Tree:  Macro Train Precision, recall, f1-score')
print ( precision_recall_fscore_support(Y_train, Y_train_pred_DT, average='macro'))
print ('Tree:  Macro Test Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_test_pred_DT, average='macro'))
print ()

print ('train-Macro-Precision-Recall-FScore',precision_recall_fscore_support(Y_train, Y_train_pred_DT, average='macro'))
print ('test-Macro-Precision-Recall-FScore',precision_recall_fscore_support(Y_test, Y_test_pred_DT, average='macro'))
print ('\n')


pr_y_test_pred_DT=clfDT.predict_proba(X_test)
print(pr_y_test_pred_DT)

#ROC curve for tree 1
fprDT, tprDT, thresholdsDT = roc_curve(Y_test, pr_y_test_pred_DT[:,1])




#ROC Curve plotting
lw=2
plt.figure(10)
plt.plot(fprDT,tprDT,color='blue')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve using DecisionTreeClassification')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()




#################################################################################


# splitting thedat sets into training and tests
X_train, X_test, Y_train, Y_test = train_test_split (features,y, test_size=size)




# defining neural network 1
clfANN1 = MLPClassifier(solver='adam', activation='relu',
                    batch_size=1, tol=1e-19,
                    hidden_layer_sizes=(25), random_state=1, max_iter=1000, verbose=False)
   
# training the neural network 1 classifier                 
clfANN1.fit(X_train, Y_train)                         

# testing the trained neural network 1 on the test set
Y_test_pred_ANN1=clfANN1.predict(X_test)

# Confusion matrixes for neural network
confMatrixTestANN1=confusion_matrix(Y_test, Y_test_pred_ANN1)

print(confMatrixTestANN1)


print ('Accuracy Test=', accuracy_score(Y_test, Y_test_pred_ANN1, normalize=True))


# measures of performance for neural network 1: Precision, Recall, F1
print ('Artificial Neural Network TEST: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_test_pred_ANN1, average='macro'))
print ('\n')
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred_ANN1)))
# ROC curve for artificial neural network 1
pr_y_test_pred_ANN1=clfANN1.predict_proba(X_test)
fprANN1, tprANN1, thresholdsANN1 = roc_curve(Y_test, pr_y_test_pred_ANN1[:,1])

#ROC Curve plotting
lw=2
plt.figure(10)
plt.plot(fprANN1,tprANN1,color='blue')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve using Artificial Neural Network')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()




#################################################################################


# defining neural network 1
clfANN2 = MLPClassifier(solver='adam', activation='relu',
                    batch_size=1, tol=1e-19,
                    hidden_layer_sizes=(5,5), random_state=1, max_iter=1000, verbose=False)
   
# training the neural network 1 classifier                 
clfANN2.fit(X_train, Y_train)                         

# testing the trained neural network 1 on the test set
Y_test_pred_ANN2=clfANN2.predict(X_test)

# Confusion matrixes for neural network
confMatrixTestANN2=confusion_matrix(Y_test, Y_test_pred_ANN2)

print(confMatrixTestANN2)


print ('Accuracy Test=', accuracy_score(Y_test, Y_test_pred_ANN2, normalize=True))


# measures of performance for neural network 1: Precision, Recall, F1
print ('Artificial Neural Network TEST: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(Y_test, Y_test_pred_ANN2, average='macro'))
print ('\n')
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred_ANN2)))
# ROC curve for artificial neural network 1
pr_y_test_pred_ANN2=clfANN2.predict_proba(X_test)
fprANN2, tprANN2, thresholdsANN2 = roc_curve(Y_test, pr_y_test_pred_ANN2[:,1])

#ROC Curve plotting
lw=2
plt.figure(10)
plt.plot(fprANN2,tprANN2,color='blue')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve using Artificial Neural Network')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()


