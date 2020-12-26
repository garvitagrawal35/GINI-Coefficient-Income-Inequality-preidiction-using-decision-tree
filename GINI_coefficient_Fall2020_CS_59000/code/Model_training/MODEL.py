#!/usr/bin/env python
# coding: utf-8

# In[388]:


# importing data library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#reading the data file
dataset = pd.read_csv("C:\\Users\\agraw\\Documents\\PNW\\Fall 2020\\ML project\\Dataset\\Final_Data\\preprocessdata_2.csv")


# In[3]:


#showing the head of the data
dataset.head()


# In[4]:


dataset.tail()


# In[5]:


#showing the dataset size
dataset.shape


# In[6]:


#preparing dependednt and independent values
dataset = dataset.drop('Date', axis=1)
x = dataset.drop('GINIALLRF' , axis=1)
y = dataset['GINIALLRF']


# In[7]:


#preprocessing the data
from sklearn.preprocessing import MinMaxScaler
X_norm = MinMaxScaler().fit_transform(x)
Y = y.to_numpy()


# In[8]:


#using lable encoder encoding the independent value
from sklearn import utils
from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
Y_encod = lab_enc.fit_transform(Y)
print(Y_encod)
print(utils.multiclass.type_of_target(Y))
print(utils.multiclass.type_of_target(Y.astype('int')))
print(utils.multiclass.type_of_target(Y_encod))


# In[9]:


#Selecting the attributes by using logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
model = LogisticRegression()
rfe = RFE(model, 12)
fit = rfe.fit(X_norm, Y_encod)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))


# In[10]:


print(fit.summary())


# In[11]:


x.head()


# In[12]:


#droping the attributes as suggested by logistic regression
data = dataset.drop(['W019RCQ027SBEA','ASCOEPQ027S','GDP','GNP','Government','HNICTIQ027S'], axis=1)
X_norm = MinMaxScaler().fit_transform(data)


# In[335]:


#preparing the test and train data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_norm, Y_encod, test_size=0.30)


# In[361]:


#training the model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "gini",max_depth=6)
classifier.fit(X_train, y_train)


# In[362]:


#Testing the model
y_pred = classifier.predict(X_test)


# In[363]:


#Checking the accuracy and error
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[364]:


from sklearn import metrics
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Accuracy:', round(metrics.accuracy_score(y_test, y_pred),3)*100)
print('Balanced Accuracy:', round(metrics.balanced_accuracy_score(y_test, y_pred),3)*100)


# In[365]:


#representing the tree
from sklearn import tree
text_representation = tree.export_text(classifier)
print(text_representation)


# In[384]:


y_test


# In[387]:


fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], lw=3)
ax.set_xlabel('Encoded actual values')
ax.set_ylabel('Encoded predicted values')
ax.set_title("Gini Coefficient Prediction Accuracy")
plt.show()


# In[385]:


Y_test


# In[357]:


fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(classifier, filled=True)


# In[396]:


import graphviz
dot_data = tree.export_graphviz(classifier, out_file=None, 
                                filled=True)
graphviz.Source(dot_data, format="png")


# In[390]:


import pylab
pylab.savefig("tree.png")


# In[ ]:





# In[ ]:




