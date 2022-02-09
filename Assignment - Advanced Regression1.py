#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement :  

#  A US-based housing company named Surprise Housing has decided to enter the Australian market. The company uses data analytics to purchase houses at a price below their actual values and flip them on at a higher price. For the same purpose, the company has collected a data set from the sale of houses in Australia.
#  
# The company is looking at prospective properties to buy to enter the market. You are required to build a regression model using regularisation in order to predict the actual value of the prospective properties and decide whether to invest in them or not

# The company wants to know:
# 
# Which variables are significant in predicting the price of a house, and
# 
# How well those variables describe the price of a house.

# ## Business Goal 
# 
# 

# 1. Required to model the price of houses with the available independent variables. 
# 2.This model will then be used by the management to understand how exactly the prices vary with the variables. 
# 3. They can accordingly manipulate the strategy of the firm and concentrate on areas that will yield high returns. 
# 
# Further, the model will be a good way for management to understand the pricing dynamics of a new market

# In[1]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importing all required packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[3]:


# Importing dataset
SHAus = pd.read_csv('C:/Users/NIKITA/Downloads/train.csv')


# In[4]:


SHAus.head()


# In[5]:


print(SHAus.info())
print(SHAus.shape)


# In[7]:


print(SHAus.isnull().any())
SHAus.describe()


# In[8]:


# checking for null values in all categorical columns

SHAus.select_dtypes(include='object').isnull().sum()[SHAus.select_dtypes(include='object').isnull().sum()>0]


# In[9]:


# Replacing NA with blank in the following columns below : 

for col in ('Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'Electrical'):
    
    SHAus[col]=SHAus[col].fillna('blank')


# In[10]:


SHAus.select_dtypes(include='object').isnull().sum()[SHAus.select_dtypes(include='object').isnull().sum()>0]


# In[11]:


# Dropping the following columns that have more than 85 percent values associated to a specific value

# Method to get the column names that have count of one value more than 85

def getHighCategoricalValueCounts():
    column = []
    categorical_columns = SHAus.select_dtypes(include=['object'])
    for col in (categorical_columns):
        if(SHAus[col].value_counts().max() >= 1241):
            column.append(col)
    return column

columnsToBeRemoved = getHighCategoricalValueCounts()

#To remove the columns with skewed data

SHAus.drop(columnsToBeRemoved, axis = 1, inplace = True)

SHAus.head()


# In[12]:


# checking the null values for the numerical data

SHAus.select_dtypes(include=['int64','float']).isnull().sum()[SHAus.select_dtypes(include=['int64','float']).isnull().sum()>0]


# In[13]:


# Impute the null values with median values for LotFrontage and MasVnrArea columns

SHAus['LotFrontage'] = SHAus['LotFrontage'].replace(np.nan, SHAus['LotFrontage'].median())
SHAus['MasVnrArea'] = SHAus['MasVnrArea'].replace(np.nan, SHAus['MasVnrArea'].median())


# In[14]:


# Setting the null values with 0 for GarageYrBlt for now as we would be handling this column further below

SHAus['GarageYrBlt']=SHAus['GarageYrBlt'].fillna(0)
SHAus['GarageYrBlt'] = SHAus['GarageYrBlt'].astype(int)


# In[15]:


# Create a new column named IsRemodelled - This column would determine whether the house has been remodelled or not based on 
# the difference between remodelled and built years

def checkForRemodel(row):
    if(row['YearBuilt'] == row['YearRemodAdd']):
        return 0
    elif(row['YearBuilt'] < row['YearRemodAdd']):
        return 1
    else:
        return 2
    
SHAus['IsRemodelled'] = SHAus.apply(checkForRemodel, axis=1)
SHAus.head()   


# In[16]:


# Create a new column named BuiltOrRemodelledAge and determine the age of the building at the time of selling

def getBuiltOrRemodelAge(row):
    if(row['YearBuilt'] == row['YearRemodAdd']):
        return row['YrSold'] - row['YearBuilt']
    else:
        return row['YrSold'] - row['YearRemodAdd']
       
SHAus['BuiltOrRemodelAge'] = SHAus.apply(getBuiltOrRemodelAge, axis=1)
SHAus.head()


# In[17]:


# Create a new column which would indicate if the Garage is old or new.
# Garage Yr Built less than 2000 will be considered as old (0) else new(1). 
# For GarageYrBuilt , where we have imputed the value as 0 will also be treated as old.

def getGarageConstructionPeriod(row):
    if row == 0:
        return 0
    elif row >= 1900 and row < 2000:        
        return 0
    else:   
        return 1
    
SHAus['OldOrNewGarage'] = SHAus['GarageYrBlt'].apply(getGarageConstructionPeriod)
SHAus.head() 


# In[18]:


# Since we have created new features from YearBuilt, YearRemodAdd, YrSold and GarageYrBlt, we can drop these columns as we 
# would only be using the derived columns for further analysis

SHAus.drop(['YearBuilt', 'YearRemodAdd', 'YrSold', 'GarageYrBlt'], axis = 1, inplace = True)


# In[19]:


# Drop the following columns that have more than 85% values associated to a specific value
# We will also drop MoSold as we will not be using that for further analysis

def getHighNumericalValueCounts():
    column = []
    numerical_columns = SHAus.select_dtypes(include=['int64', 'float'])
    for col in (numerical_columns):
        if(SHAus[col].value_counts().max() >= 1241):
            column.append(col)
    return column

columnsToBeRemoved = getHighNumericalValueCounts()
SHAus.drop(columnsToBeRemoved, axis = 1, inplace = True)

SHAus.drop(['MoSold'], axis = 1, inplace = True)

SHAus.head()


# In[20]:


# check for percentage of null values in each column

percent_missing = round(100*(SHAus.isnull().sum()/len(SHAus.index)), 2)
print(percent_missing)


# In[21]:


# Check if there are any duplicate values in the dataset

SHAus[SHAus.duplicated(keep=False)]


# In[22]:


# Checking outliers at 25%,50%,75%,90%,95% and above

SHAus.describe(percentiles=[.25,.5,.75,.90,.95,.99])


# In[23]:


# Check the outliers in all the numeric columns

plt.figure(figsize=(17, 20))
plt.subplot(5,3,1)
sns.boxplot(y = 'LotArea', palette='Set3', data = SHAus)
plt.subplot(5,3,2)
sns.boxplot(y = 'MasVnrArea', palette='Set3', data = SHAus)
plt.subplot(5,3,3)
sns.boxplot(y = 'TotalBsmtSF', palette='Set3', data = SHAus)
plt.subplot(5,3,4)
sns.boxplot(y = 'WoodDeckSF', palette='Set3', data = SHAus)
plt.subplot(5,3,5)
sns.boxplot(y = 'OpenPorchSF', palette='Set3', data = SHAus)
plt.show()


# In[24]:


# Removing Outliers

# Removing values beyond 98% for LotArea

nn_quartile_LotArea = SHAus['LotArea'].quantile(0.98)
SHAus = SHAus[SHAus["LotArea"] < nn_quartile_LotArea]


# In[25]:



# Removing values beyond 98% for MasVnrArea

nn_quartile_MasVnrArea =SHAus['MasVnrArea'].quantile(0.98)
SHAus = SHAus[SHAus["MasVnrArea"] < nn_quartile_MasVnrArea]

# Removing values beyond 99% for TotalBsmtSF

nn_quartile_TotalBsmtSF =SHAus['TotalBsmtSF'].quantile(0.99)
SHAus = SHAus[SHAus["TotalBsmtSF"] < nn_quartile_TotalBsmtSF]

# Removing values beyond 99% for WoodDeckSF

nn_quartile_WoodDeckSF = SHAus['WoodDeckSF'].quantile(0.99)
SHAus = SHAus[SHAus["WoodDeckSF"] < nn_quartile_WoodDeckSF]

# Removing values beyond 99% for OpenPorchSF

nn_quartile_OpenPorchSF = SHAus['OpenPorchSF'].quantile(0.99)
SHAus = SHAus[SHAus["OpenPorchSF"] < nn_quartile_OpenPorchSF]


# In[26]:


# Determine the percentage of data retained

num_data = round(100*(len(SHAus)/1460),2)
print(num_data)


# In[27]:


# Visualise the target variable -> SalePrice after transforming the sales price

SHAus['SalePrice'] = np.log1p(SHAus['SalePrice'])

plt.title('SalePrice')
sns.distplot(SHAus['SalePrice'], bins=10)
plt.show()


# In[28]:


# Check the numerical values using pairplots

plt.figure(figsize=(10,5))
sns.pairplot(SHAus, x_vars=['MSSubClass','LotFrontage','LotArea'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
sns.pairplot(SHAus, x_vars=['OverallQual', 'OverallCond','MasVnrArea'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
sns.pairplot(SHAus, x_vars=['BsmtFinSF1', 'BsmtUnfSF','TotalBsmtSF'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
sns.pairplot(SHAus, x_vars=['1stFlrSF','2ndFlrSF', 'GrLivArea'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
sns.pairplot(SHAus, x_vars=['BsmtFullBath','FullBath', 'HalfBath'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
sns.pairplot(SHAus, x_vars=['BedroomAbvGr','TotRmsAbvGrd', 'Fireplaces'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
sns.pairplot(SHAus, x_vars=['GarageCars','GarageArea', 'WoodDeckSF'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
sns.pairplot(SHAus, x_vars=['OpenPorchSF','SalePrice', 'IsRemodelled'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
sns.pairplot(SHAus, x_vars=['BuiltOrRemodelAge'], y_vars='SalePrice',height=4, aspect=1,kind='scatter')
plt.show()


# In[29]:


# Check the correlation of numerical columns

plt.figure(figsize = (20, 10))
sns.heatmap(SHAus.corr(), annot = True, cmap="Greens")
plt.show()


# In[30]:


# Removing the highly correlated variables

SHAus.drop(['TotRmsAbvGrd', 'GarageArea'], axis = 1, inplace = True)


# In[31]:


# Check the shape of the dataframe

SHAus.shape


# In[32]:


# Since the values of the following fields are ordered list, we shall assign values to them in sequence

# For values which can be ordered, we have given an ordered sequence value
# For values which cannot be ordered, we have categorised them into 0 and 1

SHAus['d_LotShape'] = SHAus['LotShape'].map({'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0})
SHAus['d_ExterQual'] = SHAus['ExterQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0 })
SHAus['d_BsmtQual'] = SHAus['BsmtQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
SHAus['d_BsmtExposure'] = SHAus['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0})
SHAus['d_BsmtFinType1'] = SHAus['BsmtFinType1'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1,'None': 0})

SHAus['d_HeatingQC'] = SHAus['HeatingQC'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
SHAus['d_KitchenQual'] = SHAus['KitchenQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
SHAus['d_FireplaceQu'] = SHAus['FireplaceQu'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
SHAus['d_GarageFinish'] = SHAus['GarageFinish'].map({'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0 })
SHAus['d_BldgType'] = SHAus['BldgType'].map({'Twnhs': 5, 'TwnhsE': 4, 'Duplex': 3, '2fmCon': 2, '1Fam': 1,'None': 0 })

SHAus['d_HouseStyle'] = SHAus['HouseStyle'].map({'SLvl': 8, 'SFoyer': 7, '2.5Fin': 6, '2.5Unf': 5, '2Story': 4,'1.5Fin': 3, '1.5Unf': 2, '1Story': 1, 'None': 0 })
SHAus['d_Fence'] = SHAus['Fence'].map({'GdPrv': 4, 'GdWo': 3, 'MnPrv': 2, 'MnWw': 1, 'None': 0 })
SHAus['d_LotConfig'] = SHAus['LotConfig'].map({'Inside': 5, 'Corner': 4, 'CulDSac': 3, 'FR2': 2, 'FR3': 1,'None': 0  })

SHAus['d_MasVnrType'] = SHAus['MasVnrType'].map({'BrkCmn': 1, 'BrkFace': 1, 'CBlock': 1, 'Stone': 1, 'None': 0 })
SHAus['d_SaleCondition'] = SHAus['SaleCondition'].map({'Normal': 1, 'Partial': 1, 'Abnorml': 0, 'Family': 0,'Alloca': 0, 'AdjLand': 0, 'None': 0})
SHAus.head()


# In[33]:


# drop the old columns from which the new columns were derived
# We can also drop the id column as it will not be used any more

SHAus = SHAus.drop(['Id', 'LotShape', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'HeatingQC', 
                                'KitchenQual', 'FireplaceQu', 'GarageFinish', 'BldgType', 'HouseStyle', 'Fence', 
                                'LotConfig', 'MasVnrType', 'SaleCondition'], axis=1)

SHAus.head()


# In[34]:


# For the following columns create dummies

# Creating dummies for MSZoning

d_MSZoning = pd.get_dummies(SHAus['MSZoning'], prefix='MSZoning', drop_first = True)
SHAus = pd.concat([SHAus, d_MSZoning], axis = 1)

# Creating dummies for Neighborhood

d_Neighborhood = pd.get_dummies(SHAus['Neighborhood'], prefix='Neighborhood', drop_first = True)
SHAus = pd.concat([SHAus, d_Neighborhood], axis = 1)

# Creating dummies for RoofStyle

d_RoofStyle = pd.get_dummies(SHAus['RoofStyle'], prefix='RoofStyle', drop_first = True)
SHAus = pd.concat([SHAus, d_RoofStyle], axis = 1)

# Creating dummies for Exterior1st

d_Exterior1st = pd.get_dummies(SHAus['Exterior1st'], prefix='Exterior1st', drop_first = True)
SHAus = pd.concat([SHAus, d_Exterior1st], axis = 1)

# Creating dummies for Exterior2nd

d_Exterior2nd = pd.get_dummies(SHAus['Exterior2nd'], prefix='Exterior2nd', drop_first = True)
SHAus = pd.concat([SHAus, d_Exterior2nd], axis = 1)

# Creating dummies for Foundation

d_Foundation = pd.get_dummies(SHAus['Foundation'], prefix='Foundation', drop_first = True)
SHAus = pd.concat([SHAus, d_Foundation], axis = 1)

# Creating dummies for GarageType

d_GarageType = pd.get_dummies(SHAus['GarageType'], prefix='GarageType', drop_first = True)
SHAus = pd.concat([SHAus, d_GarageType], axis = 1)

SHAus.head()


# In[35]:


# drop the below columns as we now have new columns derived from these columns

SHAus = SHAus.drop(['MSZoning', 'Neighborhood', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'Foundation', 
                                'GarageType'], axis=1)

SHAus.head()


# In[36]:


SHAus.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[37]:


data_train, data_test =train_test_split(data, train_size=0.7, random=100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[69]:


# Putting response variable to y

alpha2 = SHAus['SalePrice']
alpha2.head()


# In[70]:


# scaling the features

from sklearn.preprocessing import scale

# storing column names in cols
# scaling (the dataframe is converted to a numpy array)

cols = alpha1.columns
alpha1 = pd.DataFrame(scale(alpha1))
alpha1.columns = cols
alpha1.columns


# In[64]:


# split into train and test

import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size = 0.3, random_state=42)


# In[65]:


len(X_train.columns)


# In[80]:


lg = LinearRegression()
lg.fit(X_train, y_train)


# In[ ]:





# In[ ]:





# In[ ]:





# In[74]:


# list pf alphas

params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 
                    9.0, 10.0, 20, 50, 100, 500, 1000 ]}

ridge = Ridge()

# cross validation

folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1) 


# In[75]:


model_cv.fit(X_train, y_train)


# In[78]:


cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=200]
cv_results.head()


# In[ ]:





# In[79]:


lasso = Lasso()

# cross validation
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train, y_train) 


# In[ ]:




