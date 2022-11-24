#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import python libraries for data manipulation and visualization
import pandas as pd
import numpy as np

#import pyplot, plotly and seaborn for visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


# load the data
data = pd.read_csv('Fraud Detection.csv')
data.head()


# In[3]:


#To check the number of rows and columns
data.shape


# In[4]:


#To check data type and for any missing values in the dataset
data.info()


# In[19]:


#to change data type from float to integer
data['amount'] = data.amount.astype('int64')


# In[6]:


#descriptive analysis of the data
data.describe()


# ** due to the large volume of the data, a random sampling function was used to select a defined number of data randomly

# In[7]:


#selecting 15000 random sample from the dateset
sample_data = data.sample(15000, random_state = 42)


# In[8]:


#To check the first 5 rows of the new sample data
sample_data.head()


# In[9]:


#To reset the index of the randomly selected sample data
sample_data = sample_data.reset_index(drop = True)


# In[10]:


#To check what we have done
sample_data.head()


# In[11]:


sample_data.shape


# #### Comparing the target column(isFraud) of the original data and the sample data,To make sure that no major changes occured while selecting the sample data

# In[12]:


data.isFraud.value_counts(normalize = True)


# In[13]:


sample_data.isFraud.value_counts(normalize = True)


# In[14]:


# to find missing values
sample_data.isna().sum()


# In[15]:


#To check data type and for any missing values in the sample dataset
sample_data.info()


# In[20]:


#descriptive analysis
sample_data.describe()


# In[21]:


sample_data.amount.describe()


# In[ ]:





# ## Exploratory Data Analysis

# #### Due to limited number of features in the dataset, the EDA carried out was limited
# #### more features related to fruadulent activities can be added in the future for better analysis and modelling

# In[22]:


#value count of the online payment type column and plot
sample_data.type.value_counts()


# In[23]:


sample_data.type.value_counts().plot.bar()
plt.show


# In[24]:


# boxplot of amount, showing us the interquatiles, average and outliers
fig = px.box(sample_data, x='amount')
fig.show()


# ### to determine the stats of amount in fraudulent(1) and non-fraudulent(0) transactions

# In[25]:


sample_data[sample_data['isFraud'] == 1]['amount'].describe()


# In[26]:


sample_data[sample_data['isFraud'] == 0]['amount'].describe()


# #### observation
# it was observed that the max amount transacted(7.7m) was among the fraudulent transactions

# In[28]:


#pie plot of 'isFraud' column
labels = list(sample_data['isFraud'].value_counts().index)
values = list(sample_data['isFraud'].value_counts().values)

fig = go.Figure(data=[go.Pie(labels= labels, values= values)])
fig.show()


# #### observation
# non fraudulent transactions is 99.9% of the target column and 0.147% for fraudulent transactions, this is an inbalance dataset, it might be difficult for our model to predict and detect fraudulent transactions effectively

# ### Multivariate analysis

# In[29]:


corr = sample_data.corr()
corr


# In[30]:


sns.heatmap(corr,cmap='RdBu',vmin=-1,vmax=1,annot=True, annot_kws={'fontsize':7, 'fontweight':'bold'})
plt.figure(figsize=(10,10))


# ### Feature Engineering

# In[31]:


sample_data.head()


# In[32]:


sample_data.columns


# In[33]:


# to drop unrelevant columns or noise before starting our modelling
sample_data = sample_data.drop( ['nameOrig','nameDest'], axis=1)


# In[34]:


# to check what we have done
sample_data.head(2)


# ## One Hot Encoding

# In[35]:


# creating dummies for our categorical feature using one-hot encoding
categorical = ['type']
categorical_dummies = pd.get_dummies(sample_data[categorical])


# In[36]:


#join the dummy data with the original data
sample_data = pd.concat([sample_data,categorical_dummies], axis=1)
sample_data = sample_data.drop( ['type'], axis=1)
sample_data.head(2)


# ### Target Selection

# In[37]:


#selecting input features for our modeling 
x= sample_data.drop(['isFraud'],axis=1)
#selecting our output feature for our modeling
y= sample_data.isFraud


# In[38]:


#import the libraries we will need for modelling
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#evaluation metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")


# In[39]:


#split into training and testing sets using a 40% split ratio
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)


# In[40]:


# TODO: initialize models
LR = LogisticRegression()
KN = KNeighborsClassifier()
DC = DecisionTreeClassifier()
RF = RandomForestClassifier()


# In[41]:


#train our data
LR.fit(x_train,y_train)
#predict using test data
prediction1 = LR.predict(x_test)
#evaluation metrics
print(classification_report(y_test,prediction1))
plt.title('Confusion Matrix of the model')
sns.heatmap(confusion_matrix(y_test,prediction1), annot=True, fmt='.5g')
plt.show()


# In[42]:


#train our data
KN.fit(x_train,y_train)
#predict using test data
prediction2 = KN.predict(x_test)
#evaluation metrics
print(classification_report(y_test,prediction2))
plt.title('Confusion Matrix of the model')
sns.heatmap(confusion_matrix(y_test,prediction2), annot=True, fmt='.5g')
plt.show()


# In[43]:


#train our data
DC.fit(x_train,y_train)
#predict using test data
prediction3 = DC.predict(x_test)
#evaluation metrics
print(classification_report(y_test,prediction3))
plt.title('Confusion Matrix of the model')
sns.heatmap(confusion_matrix(y_test,prediction3), annot=True, fmt='.5g')
plt.show()


# In[44]:


#train our data
RF.fit(x_train,y_train)
#predict using test data
prediction4 = RF.predict(x_test)
#evaluation metrics
print(classification_report(y_test,prediction4))
plt.title('Confusion Matrix of the model')
sns.heatmap(confusion_matrix(y_test,prediction4), annot=True, fmt='.5g')
plt.show()


# ## Observation

# As seen above, all model showed an accuracy score of 1 or 100%, but the precision and recall value for our prediction varies, hence it is not advisable to use accuracy as a metric for evaluation

# Logistic Regression stands to be the best model to deploy as it predicted 11 fraudulent transactions out of 12 actual fraudulent transactions

# It is important to note the type 1 error(False Positives) and type 2 error (False Negatives), attention of the relevant stakeholders should be alerted to these transactions to be on the safe side (especially the type 2 error)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




