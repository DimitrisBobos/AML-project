import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn import preprocessing
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import seaborn as sb

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


laser = pd.read_csv('laser.data')
laser.shape


#number of samples
nSamples=993

# number of testing data
nTest=100

seed = 2
np.random.seed(seed)
#with first input

# Use only one feature as predictor

selFeatures = list(laser.columns.values)
featureNumber=selFeatures[0]

print('feature=', featureNumber, ' used as predictor')
laser_X = laser.iloc[:, 0].values.reshape(-1,1)

#diabetes_X, diabetes.target = unison_shuffled_copies(diabetes_X, diabetes.target)

# Split the data into training/testing sets
laser_X_train = laser_X[:-nTest]
laser_X_test = laser_X[-nTest:]

#normalize target data
#laser_target=laser.Output/np.mean(laser.Output.reshape(-1,1))




# Split the targets into training/testing sets
laser_y_train = laser.iloc[:, 4][:-nTest]
laser_y_test = laser.iloc[:, 4][-nTest:]

# Create linear regression object
regr = linear_model.LinearRegression()





# Train the model using the training sets
regr.fit(laser_X_train, laser_y_train)

# Make predictions using the testing set
laser_y_pred = regr.predict(laser_X_test)


# The coefficients
print ('Coefficients: \n', regr.coef_[0])
print  ('Intercept: \n',regr.intercept_)
# The mean squared error
print ("Mean squared error: %.2f" % mean_squared_error(laser_y_test, laser_y_pred))
# Explained variance score: 1 is perfect prediction
print ("Mean absolute error: %.2f" % mean_absolute_error(laser_y_test, laser_y_pred))
r2=r2_score(laser_y_test, laser_y_pred)
print ("R2=",r2)

vif=1/(1-r2)
print (vif)


#f-statistic
#f=f_regression( diabetes_X_test, diabetes_y_test)

# Plot outputs
plt.scatter(laser_X_test, laser_y_test,  color='black')
plt.plot(laser_X_test, laser_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())






#with second input


selFeatures = list(laser.columns.values)
featureNumber=selFeatures[1]

print('feature=', featureNumber, ' used as predictor')
laser_X = laser.iloc[:, 1].values.reshape(-1,1)

#diabetes_X, diabetes.target = unison_shuffled_copies(diabetes_X, diabetes.target)

# Split the data into training/testing sets
laser_X_train = laser_X[:-nTest]
laser_X_test = laser_X[-nTest:]

#normalize target data
#laser_target=laser.Output/np.mean(laser.Output.reshape(-1,1))




# Split the targets into training/testing sets
laser_y_train = laser.iloc[:, 4][:-nTest]
laser_y_test = laser.iloc[:, 4][-nTest:]

# Create linear regression object
regr = linear_model.LinearRegression()





# Train the model using the training sets
regr.fit(laser_X_train, laser_y_train)

# Make predictions using the testing set
laser_y_pred = regr.predict(laser_X_test)


# The coefficients
print ('Coefficients: \n', regr.coef_[0])
print  ('Intercept: \n',regr.intercept_)
# The mean squared error
print ("Mean squared error: %.2f" % mean_squared_error(laser_y_test, laser_y_pred))
# Explained variance score: 1 is perfect prediction
print ("Mean absolute error: %.2f" % mean_absolute_error(laser_y_test, laser_y_pred))
r2=r2_score(laser_y_test, laser_y_pred)
print ("R2=",r2)

vif=1/(1-r2)
print (vif)


#f-statistic
#f=f_regression( diabetes_X_test, diabetes_y_test)

# Plot outputs
plt.scatter(laser_X_test, laser_y_test,  color='black')
plt.plot(laser_X_test, laser_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())




#with third input



selFeatures = list(laser.columns.values)
featureNumber=selFeatures[2]

print('feature=', featureNumber, ' used as predictor')
laser_X = laser.iloc[:, 2].values.reshape(-1,1)

#diabetes_X, diabetes.target = unison_shuffled_copies(diabetes_X, diabetes.target)

# Split the data into training/testing sets
laser_X_train = laser_X[:-nTest]
laser_X_test = laser_X[-nTest:]

#normalize target data
#laser_target=laser.Output/np.mean(laser.Output.reshape(-1,1))




# Split the targets into training/testing sets
laser_y_train = laser.iloc[:, 4][:-nTest]
laser_y_test = laser.iloc[:, 4][-nTest:]

# Create linear regression object
regr = linear_model.LinearRegression()





# Train the model using the training sets
regr.fit(laser_X_train, laser_y_train)

# Make predictions using the testing set
laser_y_pred = regr.predict(laser_X_test)


# The coefficients
print ('Coefficients: \n', regr.coef_[0])
print  ('Intercept: \n',regr.intercept_)
# The mean squared error
print ("Mean squared error: %.2f" % mean_squared_error(laser_y_test, laser_y_pred))
# Explained variance score: 1 is perfect prediction
print ("Mean absolute error: %.2f" % mean_absolute_error(laser_y_test, laser_y_pred))
r2=r2_score(laser_y_test, laser_y_pred)
print ("R2=",r2)

vif=1/(1-r2)
print (vif)


#f-statistic
#f=f_regression( diabetes_X_test, diabetes_y_test)

# Plot outputs
plt.scatter(laser_X_test, laser_y_test,  color='black')
plt.plot(laser_X_test, laser_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())







#with fourth input



selFeatures = list(laser.columns.values)
featureNumber=selFeatures[3]

print('feature=', featureNumber, ' used as predictor')
laser_X = laser.iloc[:, 3].values.reshape(-1,1)

#diabetes_X, diabetes.target = unison_shuffled_copies(diabetes_X, diabetes.target)

# Split the data into training/testing sets
laser_X_train = laser_X[:-nTest]
laser_X_test = laser_X[-nTest:]

#normalize target data
#laser_target=laser.Output/np.mean(laser.Output.reshape(-1,1))




# Split the targets into training/testing sets
laser_y_train = laser.iloc[:, 4][:-nTest]
laser_y_test = laser.iloc[:, 4][-nTest:]

# Create linear regression object
regr = linear_model.LinearRegression()





# Train the model using the training sets
regr.fit(laser_X_train, laser_y_train)

# Make predictions using the testing set
laser_y_pred = regr.predict(laser_X_test)


# The coefficients
print ('Coefficients: \n', regr.coef_[0])
print  ('Intercept: \n',regr.intercept_)
# The mean squared error
print ("Mean squared error: %.2f" % mean_squared_error(laser_y_test, laser_y_pred))
# Explained variance score: 1 is perfect prediction
print ("Mean absolute error: %.2f" % mean_absolute_error(laser_y_test, laser_y_pred))
r2=r2_score(laser_y_test, laser_y_pred)

print ("R2=",r2)

vif=1/(1-r2)
print (vif)


#f-statistic
#f=f_regression( diabetes_X_test, diabetes_y_test)

# Plot outputs
plt.scatter(laser_X_test, laser_y_test,  color='black')
plt.plot(laser_X_test, laser_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())















####################################################################################
 #using all the features

#dataframe for independent variables
X = laser.iloc[:, 0: 4]
heat_map = sb.heatmap(X)
plt.show()

#dataframe for dependent variables
y = laser.iloc[:, 4]

 

#split to test and train data

 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

 

#Linear Regression
lreg = linear_model.LinearRegression()
 

lreg.fit(X_train, y_train)

 

ep_predict = lreg.predict(X_test)

 

#Model evaluation
print("'Coefficients: \n ", lreg.coef_)
print('Intercept: \n',lreg.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, ep_predict))
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, ep_predict))
r2=r2_score(y_test, ep_predict)
print("R2=", r2_score(y_test, ep_predict))
vif=1/(1-r2)
print (vif)




####################################################################################
#Lasso and Ridge




regr2 = linear_model.LinearRegression()
regr3 = linear_model.LinearRegression()
regr3 = Ridge(alpha=0.4, normalize=True)
regr2=Lasso(alpha=0.5, normalize=True)
regr2.fit(X_train, y_train)
regr3.fit(X_train, y_train)



# Make predictions using the testing set, linear
df_y_pred2 = regr2.predict(X_test)
df_y_pred3 = regr3.predict(X_test)



# The coefficients
print ('Coefficients Linear Regression Lasso: \n', regr2.coef_)
print ('Interecept: ',regr2.intercept_)
# The mean squared error
"Mean squared error: %.2f" % mean_squared_error(y_test, df_y_pred2)
# Explained variance score: 1 is perfect prediction
print ('R2 score Linear: %.2f' % r2_score(y_test, df_y_pred2))
print ("Mean absolute error Linear: %.2f" % mean_absolute_error(y_test, df_y_pred2))
print ("Mean squared error Linear: %.2f" % mean_squared_error(y_test, df_y_pred2))
r2 = r2_score(y_test, df_y_pred2)
vif=1/(1-r2)
print (vif)




# The coefficients
print ('Coefficients Linear Regression Ridge: \n', regr3.coef_)
print ('Interecept: ',regr3.intercept_)
# The mean squared error
"Mean squared error: %.2f" % mean_squared_error(y_test, df_y_pred3)
# Explained variance score: 1 is perfect prediction
print ('R2 score Linear: %.2f' % r2_score(y_test, df_y_pred3))
print ("Mean absolute error Linear: %.2f" % mean_absolute_error(y_test, df_y_pred3))
print ("Mean squared error Linear: %.2f" % mean_squared_error(y_test, df_y_pred3))
r2 = r2_score(y_test, df_y_pred3)
vif=1/(1-r2)
print (vif)


















####################################################################################
#Polynomial








lregPoly = linear_model.LinearRegression()

poly = PolynomialFeatures(degree=2)
Xpoly=poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)





lregPoly.fit(X_train, y_train)
predP = lregPoly.predict(X_test)




#calculating mse


mseP = np.mean((predP - y_test)**2)
print ('mse polynomial',mseP)
print ('R2 Poly ',r2_score(y_test, predP))

#r2_score(x_cv, y_cv)
#r2_score(x_cvP, y_cvP)


# The coefficients
print ('Coefficients Linear Regression: \n', lregPoly.coef_)
print ('Interecept: ',lregPoly.intercept_)
# The mean squared error
"Mean squared error: %.2f" % mean_squared_error(y_test, predP)
# Explained variance score: 1 is perfect prediction
print ('R2 score Linear: %.2f' % r2_score(y_test, predP))
print ("Mean absolute error Linear: %.2f" % mean_absolute_error(y_test, predP))
print ("Mean squared error Linear: %.2f" % mean_squared_error(y_test, predP))
r2 = r2_score(y_test, predP)
vif=1/(1-r2)
print (vif)








####################################################################################


pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3' ])

finalDf = pd.concat([principalDf, laser.iloc[:, 4]], axis = 1)


print(pca.explained_variance_ratio_)



#dataframe for independent variables
X = finalDf.iloc[:, 0: 3]

 

#dataframe for dependent variables
y = finalDf.iloc[:, 3]

 

#split to test and train data

 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

 

#Linear Regression
lreg = linear_model.LinearRegression()
 

lreg.fit(X_train, y_train)

 

ep_predict = lreg.predict(X_test)

 

#Model evaluation
print("'Coefficients: \n ", lreg.coef_)
print('Intercept: \n',lreg.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, ep_predict))
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, ep_predict))
print("R2=", r2_score(y_test, ep_predict))
vif=1/(1-r2)
print (vif)



