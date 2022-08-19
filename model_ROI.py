#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
from mlxtend.preprocessing import minmax_scaling
from scipy import stats as st
#Plotting
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# In[2]:


df = pd.read_csv("prosperLoanData.csv")


# In[3]:


#Remove outstanding loans
df = df[df["LoanStatus"] != "Current"]
df["LoanStatus"].value_counts()


# In[4]:


#Encode all completed loans as 1, and all delinquent, chargedoff, cancelled and defaulted loans as 0
df["LoanStatus"] = (df["LoanStatus"] == "Completed").astype(int)
df.drop(["ListingKey", "ListingNumber", "LoanKey", "LoanNumber",'LoanFirstDefaultedCycleNumber',"MemberKey","GroupKey"], axis=1, inplace=True)


# In[5]:


categorical = df.select_dtypes(include=['bool','object']).columns
numerical=df.select_dtypes('number').columns
df_c = df[categorical].copy()
df_n = df[numerical].copy()


# In[6]:


#numerical data handling with mean
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp.fit(df_n)
df_num_imputed = imp.transform(df_n)

#categorical data with mode
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
imp.fit(df_c)
df_cat_imputed = imp.transform(df_c)


# In[7]:


#concat c and n data
df_c= pd.DataFrame(df_cat_imputed, columns=df_c.columns.tolist())
df_n= pd.DataFrame(df_num_imputed, columns=df_n.columns.tolist())
data=pd.concat([df_n,df_c],axis=1)


# In[8]:


from sklearn.preprocessing import LabelEncoder
cat_list = []
num_list = []
for colname, colvalue in df_c.iteritems():
        cat_list.append(colname)
for col in cat_list:
    encoder = LabelEncoder()
    encoder.fit(df_c[col])
    df_c[col] = encoder.transform(df_c[col])

L=df_c.columns.to_list()
df_c = pd.DataFrame(df_c, columns=L)
data=pd.concat([df_c,data],axis=1)

data = data.select_dtypes(exclude=['object'])


# In[9]:


X = data.copy()
y = df[("BorrowerRate")]


# In[10]:


X=X.loc[:,['DebtToIncomeRatio', 'StatedMonthlyIncome', 'LoanOriginalAmount',
       'MonthlyLoanPayment', 'LP_CustomerPayments',
       'LP_CustomerPrincipalPayments',
       'LP_GrossPrincipalLoss']]


# In[13]:


from sklearn.model_selection import train_test_split
# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=int(len(X) * 0.67),random_state=42)


# In[14]:


#Building the Decision Tree Model on our dataset
from sklearn.tree import DecisionTreeRegressor
DT_model = DecisionTreeRegressor(max_depth=5).fit(X_train,y_train)

#predicting the rate of interest using Decision Tree model
DT_predict = DT_model.predict(X_test)
print(DT_predict)


# In[15]:


df["LoanStatus"]


# In[16]:


print(df.iloc[2])


# In[17]:


import pickle


# In[18]:


filename='model_ROI.pkl'


# In[19]:


pickle.dump(DT_model, open(filename, 'wb'))


# In[ ]:




