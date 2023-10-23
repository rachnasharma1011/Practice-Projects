#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[3]:


insurance_data=pd.read_csv(r"C:\Users\asus 1\Desktop\Fliprobo\medical_cost_insurance.csv")


# In[4]:


insurance_data


# In[5]:


insurance_data.head()


# In[6]:


insurance_data.tail()


# In[7]:


insurance_data.shape


# In[8]:


insurance_data.info()


# In[9]:


insurance_data.isnull().sum()


# In[10]:


insurance_data.columns


# In[12]:


insurance_data.describe()


# In[16]:


plt.figure(figsize=(5,5))
sns.countplot(x='sex', data=insurance_data)
plt.show()


# In[17]:


insurance_data['sex'].value_counts()


# In[18]:


sns.countplot(x='smoker', data=insurance_data)
plt.show()


# In[22]:


sns.countplot(x='region', data=insurance_data)
plt.show()


# In[23]:


sns.barplot(x='region', y='charges', data=insurance_data)
plt.title('Cost vs Region')
plt.show()


# In[24]:


sns.barplot(x='smoker', y='charges', data=insurance_data)
plt.title('Cost vs smokers')
plt.show()


# In[25]:


sns.barplot(x='sex', y='charges', hue='smoker', data=insurance_data)
plt.title('Charges for smokers')
plt.show()


# In[26]:


#male smokers have high medical charges


# In[32]:


fig, axes= plt.subplots(1, 3, figsize=(8, 5), sharey=True)
fig.suptitle('Visualizing Categorical Columns')
sns.barplot(x='smoker', y='charges', data=insurance_data,  ax=axes[0])
sns.barplot(x='sex', y='charges', data=insurance_data, ax=axes[1])
sns.barplot(x='region', y='charges', data=insurance_data, ax=axes[2])
plt.show()


# In[36]:


insurance_data[['age', 'bmi', 'children', 'charges']].hist(bins=30, figsize=(10,10), color='red')
plt.show()


# In[37]:


insurance_data['sex']=insurance_data['sex'].apply({'male':0, 'female':1}.get)
insurance_data['smoker']=insurance_data['smoker'].apply({'no':0, 'yes':1}.get)
insurance_data['region']=insurance_data['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)


# In[38]:


insurance_data.head()


# In[40]:


plt.figure(figsize=(10,7))
sns.heatmap(insurance_data.corr(), annot=True)
plt.show()


# In[41]:


#the heatmap shows that habit of smoking affect the insurance cost and there is least effect of gender on the insurance cost


# In[42]:


X=insurance_data.drop(['charges', 'sex'], axis=1)
y=insurance_data['charges']


# In[43]:


X


# In[44]:


y


# In[46]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=17)


# In[48]:


print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)
print('y_train shape: ', y_train.shape)
print('y_test shape: ', y_test.shape)


# In[50]:


linreg=LinearRegression()
linreg.fit(X_train, y_train)
pred=linreg.predict(X_test)


# In[51]:


from sklearn.metrics import r2_score


# In[52]:


print('R2 score : ', (r2_score(y_test, pred)))


# # Prediction of insurance cost

# In[55]:


index=[0]
sample_data=pd.DataFrame({'age':50, 'bmi':25, 'children':2, 'smoker':1, 'region':2}, index)
sample_data


# In[58]:


sample_pred=linreg.predict(sample_data)
print('The predicted medical insurance cost is ', sample_pred)


# In[59]:


# lets find the insurance cost for non smaker with the same data
index=[0]
sample_data1=pd.DataFrame({'age':50, 'bmi':25, 'children':2, 'smoker':0, 'region':2}, index)
sample_data1


# In[60]:


sample_pred1=linreg.predict(sample_data1)
print('The predicted medical insurance cost is ', sample_pred1)


# In[ ]:


#We can conclude that insurance cost become very high for smokers 

