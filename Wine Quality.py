#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[43]:


wine_data =pd.read_csv(r'C:\Users\asus 1\Desktop\Fliprobo\winequality-red.csv')
wine_data


# In[3]:


wine_data.head()


# # Exploratory Data Analysis

# In[4]:


wine_data.shape


# In[5]:


wine_data.info()


# In[6]:


wine_data.describe()


# In[7]:


wine_data['quality'].unique()


# In[8]:


wine_data.isna().sum()


# In[10]:


fig=plt.figure(figsize=(5,4))
plt.bar(wine_data['quality'], wine_data['fixed acidity'])
plt.xlabel('Quality')
plt.ylabel('Fixed Acidity')
plt.show()


# In[11]:


wine_data['quality'].value_counts()


# In[12]:


sns.countplot(wine_data['quality'])
plt.show()


# In[13]:


fig=plt.figure(figsize=(5,4))
plt.bar(wine_data['quality'], wine_data['volatile acidity'])
plt.xlabel('Quality')
plt.ylabel('Volatile Acidity')
plt.show()
# with decrease in volatile acidity the quality of wine increases, so volatile acidity is inversely proportional to quality


# In[14]:


wine_data.hist(bins=100, figsize=(12, 12))
plt.show()


# In[15]:


fig=plt.figure(figsize=(5,4))
plt.bar(wine_data['quality'], wine_data['citric acid'])
plt.xlabel('Quality')
plt.ylabel('Citric Acid')
plt.show()


# In[16]:


fig=plt.figure(figsize=(5,4))
plt.bar(wine_data['quality'], wine_data['residual sugar'])
plt.xlabel('Quality')
plt.ylabel('Residual Sugar')
plt.show()


# In[17]:


fig=plt.figure(figsize=(5,4))
plt.bar(wine_data['quality'], wine_data['chlorides'])
plt.xlabel('Quality')
plt.ylabel('Chlorides')
plt.show()
# with decrease in chlorides the quality of wine increases


# In[18]:


fig=plt.figure(figsize=(5,4))
plt.bar(wine_data['quality'], wine_data['free sulfur dioxide'])
plt.xlabel('Quality')
plt.ylabel('Free Sulfur Dioxide')
plt.show()


# In[19]:


fig=plt.figure(figsize=(5,4))
plt.bar(wine_data['quality'], wine_data['total sulfur dioxide'])
plt.xlabel('Quality')
plt.ylabel('Total Sulfur Dioxide')
plt.show()
# total sulfur dioxide has not much corelation with the quality of wine


# In[20]:


fig=plt.figure(figsize=(5,4))
plt.bar(wine_data['quality'], wine_data['density'])
plt.xlabel('Quality')
plt.ylabel('Density')
plt.show()


# In[21]:


fig=plt.figure(figsize=(5,4))
plt.bar(wine_data['quality'], wine_data['pH'])
plt.xlabel('Quality')
plt.ylabel('pH')
plt.show()


# In[22]:


fig=plt.figure(figsize=(5,4))
plt.bar(wine_data['quality'], wine_data['sulphates'])
plt.xlabel('Quality')
plt.ylabel('sulphates')
plt.show()


# In[23]:


fig=plt.figure(figsize=(5,4))
plt.bar(wine_data['quality'], wine_data['alcohol'])
plt.xlabel('Quality')
plt.ylabel('alcohol')
plt.show()
# quality of wine is directly proportional to alcohol content


# # Correlation Matrix

# In[24]:


wine_data.corr()


# In[25]:


plt.figure(figsize=(10,8))
sns.heatmap(wine_data.corr(), annot=True)
plt.title('Corelation between different columns')
plt.show()


# # alcohol has highest and volatile acidity has lowest correlation with the quality of wine

# In[26]:


wine_data.corr()['quality'].sort_values()


# # Data Processing

# In[44]:


wine_data['quality']=wine_data.quality.apply(lambda x:1 if x>=7 else 0)


# In[45]:


wine_data['quality'].value_counts()


# In[46]:


sns.countplot(wine_data['quality'])


# In[47]:


X=wine_data.drop('quality', axis=1)
X


# In[49]:


y=wine_data['quality']
y


# In[50]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=17)


# In[51]:


print("X_train :", X_train.shape)
print("X_test :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test :", y_test.shape)


# # Model Training

# # Logistic Regression

# In[54]:


lr=LogisticRegression()


# In[55]:


lr.fit(X_train, y_train)


# In[56]:


y_pred_lr=lr.predict(X_test)


# In[57]:


accuracy_lr=accuracy_score(y_test, y_pred_lr)
accuracy_lr


# In[60]:


classification_report_lr=classification_report(y_test, y_pred_lr)
print(classification_report_lr)


# In[61]:


confusion_matrix_lr=confusion_matrix(y_test, y_pred_lr)
confusion_matrix_lr


# In[62]:


print("test accuracy is: {:.2f}%".format(accuracy_lr*100))


# # Decision Tree model

# In[64]:


dtree=DecisionTreeClassifier()
dtree.fit(X_train, y_train)
y_pred_dtree=dtree.predict(X_test)
accuracy_dtree=accuracy_score(y_test, y_pred_dtree)
print("test accuracy is: {:.2f}%".format(accuracy_dtree*100))


# In[66]:


classification_report_dtree=classification_report(y_test, y_pred_dtree)
print(classification_report_dtree)


# In[65]:


confusion_matrix_dtree=confusion_matrix(y_test, y_pred_dtree)
confusion_matrix_dtree


# # Extra Tree Classifier

# In[67]:


from sklearn.ensemble import ExtraTreesClassifier


# In[68]:


etc=ExtraTreesClassifier()


# In[69]:


etc.fit(X_train, y_train)


# In[71]:


y_pred_etc=etc.predict(X_test)


# In[73]:


accuracy_score_etc=accuracy_score(y_test, y_pred_etc)
print("test accuracy is: {:.2f}%".format(accuracy_score_etc*100))


# In[74]:


confusion_matrix_etc=confusion_matrix(y_test, y_pred_etc)
print(confusion_matrix_etc)


# In[75]:


classification_report_etc=classification_report(y_test, y_pred_etc)
print(classification_report_etc)


# # Random Forest Classifier

# In[76]:


rfc=RandomForestClassifier()


# In[77]:


rfc.fit(X_train, y_train)


# In[78]:


y_pred_rfc=rfc.predict(X_test)


# In[79]:


accuracy_score_rfc=accuracy_score(y_test, y_pred_rfc)
print("test accuracy is: {:.2f}%".format(accuracy_score_rfc*100))


# In[82]:


confusion_matrix_rfc=confusion_matrix(y_test, y_pred_rfc)
print(confusion_matrix_rfc)


# In[83]:


classification_report_rfc=classification_report(y_test, y_pred_rfc)
print(classification_report_rfc)


# In[85]:


#Random forest has better accuracy than other models


# # Prediction

# In[87]:


input_data=(7.8, 0.76, 0.04, 2.3, 0.092, 15.0, 54.0, 0.9970, 3.26, 0.65, 9.8)
np_input=np.asarray(input_data)
np_input_reshaped=np_input.reshape(1, -1)

prediction=rfc.predict(np_input_reshaped)
print(prediction)

