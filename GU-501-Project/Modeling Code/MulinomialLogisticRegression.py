# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:09:35 2019

@author: VChan
"""

#################################################################################################################
subsetdrugdensityonlycol['FEMALE_PER'] = round(subsetdrugdensityonlycol['TOT_FEMALE']/subsetdrugdensityonlycol['TOT_POP'],2)
subsetdrugdensityonlycol['TOT_MALE'] = round(subsetdrugdensityonlycol['TOT_MALE']/subsetdrugdensityonlycol['TOT_POP'],2)
subsetdrugdensityonlycol['HSorLess'] = subsetdrugdensityonlycol['ltHS'] + subsetdrugdensityonlycol['HS']
subsetdrugdensityonlycol['CollegeorMore'] =subsetdrugdensityonlycol['sCorA'] + subsetdrugdensityonlycol['B+']
################################################################################################################


### Logistic REGRESSION WITH THE FIRST MODEL : (CONVERGENCE WARNING)
from sklearn import linear_model
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# Instantiate model
lm1 = LogisticRegression()

feature_cols1=['WA_PER','Poverty_Percent','crimerate',
                                           'Median_Income','B+','ObesityPercent','democrat','Unemp_Yearly_Rate']
X1 = subsetdrugdensityonlycol[feature_cols1] # features
y1 = subsetdrugdensityonlycol.densitybins # Target variable

# Split data
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2, random_state=1)

mul_lr1 = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X1_train, y1_train)

print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(y_train, mul_lr1.predict(X1_train)))
print("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(y_test, mul_lr1.predict(X1_test)))



## print coefficients
#print(lm1.intercept_)
#print(lm1.coef_)
#
## pair the feature names with the coefficients
#list(zip(feature_cols, mul_lr1.coef_))
#
## Predict
#y_pred = lm1.predict(X_test)
#
## RMSE
#print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


#### Second Regression Model (CONVERGENCE WARNING)

feature_cols2 = ['HS','Median_Income','WA_PER']

X2 = subsetdrugdensityonlycol[feature_cols2] # features
y2 = subsetdrugdensityonlycol.densitybins # Target variable

# Split data
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.2, random_state=1)

# Instantiate model
lm2 = LogisticRegression()

mul_lr2 = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X2_train, y2_train)


print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(y2_train, mul_lr2.predict(X2_train)))
print("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(y2_test, mul_lr2.predict(X2_test)))



#### THIRD Regression Model

subsetdrugdensityonlycol['FEMALE_PER'] = round(subsetdrugdensityonlycol['TOT_FEMALE']/subsetdrugdensityonlycol['TOT_POP'],2)
subsetdrugdensityonlycol['HSorLess'] = subsetdrugdensityonlycol['ltHS'] + subsetdrugdensityonlycol['HS']
subsetdrugdensityonlycol['CollegeorMore'] =subsetdrugdensityonlycol['sCorA'] + subsetdrugdensityonlycol['B+']


feature_cols3 = ['CollegeorMore','FEMALE_PER']

X3 = subsetdrugdensityonlycol[feature_cols3] # features
y3 = subsetdrugdensityonlycol.densitybins # Target variable

# Split data
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size = 0.2, random_state=1)

# Instantiate model
lm3 = LogisticRegression()

mul_lr3 = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X3_train, y3_train)


print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(y3_train, mul_lr3.predict(X3_train)))
print("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(y3_test, mul_lr3.predict(X3_test)))


#### FOURTH Regression Model


feature_cols4 = ['CollegeorMore','FEMALE_PER','Unemp_Yearly_Rate','republican']

X4 = subsetdrugdensityonlycol[feature_cols4] # features
y4 = subsetdrugdensityonlycol.densitybins # Target variable

# Split data
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size = 0.2, random_state=1)

# Instantiate model
lm4 = LogisticRegression()

mul_lr4 = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X4_train, y4_train)


print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(y4_train, mul_lr4.predict(X4_train)))
print("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(y4_test, mul_lr4.predict(X4_test)))




#### FIFTH Regression Model
feature_cols5 = ['CollegeorMore','FEMALE_PER','Unemp_Yearly_Rate','republican','WA_PER','crimerate']

X5 = subsetdrugdensityonlycol[feature_cols5] # features
y5 = subsetdrugdensityonlycol.densitybins # Target variable

# Split data
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size = 0.2, random_state=1)

# Instantiate model
lm5 = LogisticRegression()

mul_lr5 = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X5_train, y5_train)


print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(y5_train, mul_lr5.predict(X5_train)))
print("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(y5_test, mul_lr5.predict(X5_test)))





#### SIXTH Regression Model
feature_cols6 = ['CollegeorMore','FEMALE_PER','Unemp_Yearly_Rate','republican','WA_PER','crimerate','ObesityPercent']

X6 = subsetdrugdensityonlycol[feature_cols6] # features
y6 = subsetdrugdensityonlycol.densitybins # Target variable

# Split data
X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y6, test_size = 0.2, random_state=1)

# Instantiate model
lm6 = LogisticRegression()

mul_lr6 = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X6_train, y6_train)


print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(y6_train, mul_lr6.predict(X6_train)))
print("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(y6_test, mul_lr6.predict(X6_test)))




#### SEVENTH Regression Model
feature_cols7 = ['CollegeorMore','FEMALE_PER','Unemp_Yearly_Rate','republican',
                 'WA_PER','crimerate','ObesityPercent','Poverty_Percent']

X7 = subsetdrugdensityonlycol[feature_cols7] # features
y7 = subsetdrugdensityonlycol.densitybins # Target variable

# Split data
X7_train, X7_test, y7_train, y7_test = train_test_split(X7, y7, test_size = 0.2, random_state=1)

# Instantiate model
lm7 = LogisticRegression()

mul_lr7 = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X7_train, y7_train)


print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(y7_train, mul_lr7.predict(X7_train)))
print("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(y7_test, mul_lr7.predict(X7_test)))



#### EIGHTH Regression Model
feature_cols8 = ['CollegeorMore','FEMALE_PER','Unemp_Yearly_Rate','republican',
                 'WA_PER','crimerate','ObesityPercent','Poverty_Percent', 'H_PER']

X8 = subsetdrugdensityonlycol[feature_cols8] # features
y8 = subsetdrugdensityonlycol.densitybins # Target variable

# Split data
X8_train, X8_test, y8_train, y8_test = train_test_split(X8, y8, test_size = 0.2, random_state=1)

# Instantiate model
lm8 = LogisticRegression()

mul_lr8 = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X8_train, y8_train)


print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(y8_train, mul_lr8.predict(X8_train)))
print("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(y8_test, mul_lr8.predict(X8_test)))




###################### Putting Everything Together:  ##########################
reg_acc1 = metrics.accuracy_score(y1_test, mul_lr1.predict(X1_test))
reg_acc2 = metrics.accuracy_score(y2_test, mul_lr2.predict(X2_test))
reg_acc3 = metrics.accuracy_score(y3_test, mul_lr3.predict(X3_test))
reg_acc4 = metrics.accuracy_score(y4_test, mul_lr4.predict(X4_test))
reg_acc5 = metrics.accuracy_score(y5_test, mul_lr5.predict(X5_test))
reg_acc6 = metrics.accuracy_score(y6_test, mul_lr6.predict(X6_test))
reg_acc7 = metrics.accuracy_score(y7_test, mul_lr7.predict(X7_test))
reg_acc8 = metrics.accuracy_score(y8_test, mul_lr8.predict(X8_test))


Trial = [1,2,3,4,5,6,7,8]
Accuracy = [reg_acc1,reg_acc2,reg_acc3,reg_acc4,reg_acc5,reg_acc6,reg_acc7,reg_acc8]
plt.plot(Trial,Accuracy, '-ok')

plt.xlabel('Trial')
plt.ylabel('Accuracy')
plt.title('Multinomial Logistic Regression Accuracy Rate')

plt.show()

#### UNDERSTANDING COEFFICIENTS FOR EACH REGRESSION MODEL: 
list(zip(mul_lr1.coef_)) # Had convergence warning
list(zip(mul_lr2.coef_)) # Had convergence warning
list(zip(mul_lr3.coef_))
list(zip(mul_lr4.coef_))
list(zip(mul_lr5.coef_))
list(zip(mul_lr6.coef_))
list(zip(mul_lr7.coef_)) # BEST MODEL
list(zip(mul_lr8.coef_))

# Summary Model for the 7th Model:
import statsmodels.api as st
mdl7 = st.MNLogit(y7_train, X7_train)
mdl7_fit = mdl7.fit()
print(mdl7_fit.summary())