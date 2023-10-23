#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


insurance_data=pd.read_csv(r"C:\Users\asus 1\Desktop\Fliprobo\medical_cost_insurance.csv")


# In[3]:


insurance_data


# In[4]:


insurance_data.head()


# In[5]:


insurance_data.tail()


# In[6]:


insurance_data.shape


# In[7]:


insurance_data.info()


# In[8]:


insurance_data.isnull().sum()


# In[9]:


insurance_data.columns


# In[10]:


insurance_data.describe()


# In[11]:


plt.figure(figsize=(5,5))
sns.countplot(x='sex', data=insurance_data)
plt.show()


# In[12]:


insurance_data['sex'].value_counts()


# In[13]:


sns.countplot(x='smoker', data=insurance_data)
plt.show()


# In[14]:


sns.countplot(x='region', data=insurance_data)
plt.show()


# In[15]:


sns.barplot(x='region', y='charges', data=insurance_data)
plt.title('Cost vs Region')
plt.show()


# In[16]:


sns.barplot(x='smoker', y='charges', data=insurance_data)
plt.title('Cost vs smokers')
plt.show()


# In[17]:


sns.barplot(x='sex', y='charges', hue='smoker', data=insurance_data)
plt.title('Charges for smokers')
plt.show()


# In[18]:


#male smokers have high medical charges


# In[19]:


fig, axes= plt.subplots(1, 3, figsize=(8, 5), sharey=True)
fig.suptitle('Visualizing Categorical Columns')
sns.barplot(x='smoker', y='charges', data=insurance_data,  ax=axes[0])
sns.barplot(x='sex', y='charges', data=insurance_data, ax=axes[1])
sns.barplot(x='region', y='charges', data=insurance_data, ax=axes[2])
plt.show()


# In[20]:


insurance_data[['age', 'bmi', 'children', 'charges']].hist(bins=30, figsize=(10,10), color='red')
plt.show()


# In[21]:


insurance_data['sex']=insurance_data['sex'].apply({'male':0, 'female':1}.get)
insurance_data['smoker']=insurance_data['smoker'].apply({'no':0, 'yes':1}.get)
insurance_data['region']=insurance_data['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)


# In[22]:


insurance_data.head()


# In[23]:


plt.figure(figsize=(10,7))
sns.heatmap(insurance_data.corr(), annot=True)
plt.show()


# In[24]:


#the heatmap shows that habit of smoking affect the insurance cost and there is least effect of gender on the insurance cost


# In[25]:


X=insurance_data.drop(['charges', 'sex'], axis=1)
y=insurance_data['charges']


# In[26]:


X


# In[27]:


y


# In[28]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=17)


# In[29]:


print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)
print('y_train shape: ', y_train.shape)
print('y_test shape: ', y_test.shape)


# In[30]:


linreg=LinearRegression()
linreg.fit(X_train, y_train)
pred=linreg.predict(X_test)


# In[31]:


from sklearn.metrics import r2_score


# In[32]:


print('R2 score : ', (r2_score(y_test, pred)))


# # Decision Tree Regression

# In[39]:


from sklearn.tree import DecisionTreeRegressor


# In[42]:


dec_model=DecisionTreeRegressor()
dec_model.fit(X_train, y_train)
dec_pred=dec_model.score(X_test, y_test)
dec_pred


# # Random Forest Regression

# In[52]:


from sklearn.ensemble import RandomForestRegressor


# In[53]:


random_model=RandomForestRegressor()
random_model.fit(X_train, y_train)
random_pred=random_model.score(X_test, y_test)
random_pred


# # XG Boost Regression

# In[55]:


get_ipython().system('pip install xgboost')


# In[56]:


from xgboost import XGBRegressor


# In[58]:


xgb_model=XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_pred=xgb_model.score(X_test, y_test)
xgb_pred


# # Prediction of insurance cost

# In[59]:


index=[0]
sample_data=pd.DataFrame({'age':50, 'bmi':25, 'children':2, 'smoker':1, 'region':2}, index)
sample_data


# # Prediction using LinearRegression

# In[63]:


sample_pred=linreg.predict(sample_data)
print('The predicted medical insurance cost is ', sample_pred)


# In[64]:


# lets find the insurance cost for non smaker with the same data
index=[0]
sample_data1=pd.DataFrame({'age':50, 'bmi':25, 'children':2, 'smoker':0, 'region':2}, index)
sample_data1


# In[36]:


sample_pred1=linreg.predict(sample_data1)
print('The predicted medical insurance cost is ', sample_pred1)


# # Prediction using RandomForest

# In[61]:


prediction= random_model.predict(sample_data)
prediction


# In[65]:


prediction1= random_model.predict(sample_data1)
prediction1


# In[37]:


#We can conclude that insurance cost become very high for smokers 

