# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:09:29 2019

@author: Doug Neumann
"""
import pandas as pd 

finaldrugdata1 = pd.read_csv('subsetdrugdensityonlycol.csv')
finaldrugdata1.head(1)

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.externals.six import StringIO   
from sklearn.tree import export_graphviz
import pydotplus


# changes density bins into category instead of integer
finaldrugdata1["densitybins"] = finaldrugdata1["densitybins"].astype('category')

feature_cols = ['ltHS','HS','sCorA','B+','WA_PER']
### this was the best model run, multiple models were done for analysis 
### the different trial accuracies are listed in a spread sheet in the docs. 

X = finaldrugdata1[feature_cols] # features
y = finaldrugdata1.densitybins # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy of Tree:",metrics.accuracy_score(y_test, y_pred))




#################################### Demo #####################################
demodata = {'ltHS':[35,35,5,5,14], 'HS':[40,40,15,15,23], 'sCorA':[15,15,30,30,27],
            'B+':[10,10,50,50,26], 'WA_PER':[20,80,20,80,67]} 
  
# Create DataFrame 
demodf = pd.DataFrame(demodata) 

print(demodf)

clf.predict(demodf)



############################## Tree Visualization #############################
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols, class_names=['Bin 1','Bin 2', 'Bin 3', 'Bin 4'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree.png')

