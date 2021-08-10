#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing packages
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss


# In[2]:


data = pd.read_csv("C:\\Users\\satvi\\Downloads\\CleanedMentalHealthinTech.csv")


# In[3]:


#Obtaining a list of categorical independent variables
Xfeatures = []
for column in data:
    Xfeatures.append(str(column))

Xfeatures.remove("fep_mh_willing")
Xfeatures.remove("sought_treat")

y_vars = ["fep_mh_willing", "sought_treat"]


# In[4]:


#Finding associaiton of each variable to "Would you bring up a mental health issue with a potential employer in an interview?
print("Feature correlations")
#Calculating Cramer's V (gives a measure of association between 0 and 1)
for feature in Xfeatures:
    confusion_matrix = pd.crosstab(data[feature], data['fep_mh_willing'])
    cm = confusion_matrix.to_numpy()
    X2 = ss.chi2_contingency(cm, correction=False)[0]
    n = np.sum(cm)
    minDim = min(cm.shape)-1
    V = np.sqrt((X2/n) / minDim)
    print(str(feature) + ": " + str(V))


# In[5]:


#Finding associaiton of each variable to "Would you seek treatment?"
print("Feature correlations")
#Calculating Cramer's V (gives a measure of association between 0 and 1)
corrs = {}
for feature in Xfeatures:
    confusion_matrix = pd.crosstab(data[feature], data['sought_treat'])
    cm = confusion_matrix.to_numpy()
    X2 = ss.chi2_contingency(cm, correction=False)[0]
    n = np.sum(cm)
    minDim = min(cm.shape)-1
    V = np.sqrt((X2/n) / minDim)
    #print(str(feature) + ": " + str(V))
    corrs[feature] = V

sort_orders = sorted(corrs.items(), key=lambda x: x[1], reverse=True)
print(sort_orders)


# In[ ]:





# In[6]:


#Introducing a constructed feature (mh_discuss_office): is an individual willing to discuss mhd in the office place?
mh_discuss_office = []
for rownumber in range(0, len(data.index)):
    if data.loc[rownumber, 'fep_mh_willing'] == 1 or data.loc[rownumber, 'pep_comf_cw'] == 1 or data.loc[rownumber, 'pep_comf_sup'] == 1:
        mh_discuss_office.append(1)
    else:
        if data.loc[rownumber, 'fep_mh_willing'] == 2 or data.loc[rownumber, 'pep_comf_cw'] == 2 or data.loc[rownumber, 'pep_comf_sup'] == 2 or data.loc[rownumber, 'pep_comf_sup'] == 3:
            mh_discuss_office.append(2)
        else:
            mh_discuss_office.append(3)
    
data['mh_discuss_office'] = mh_discuss_office
  



# In[9]:


Xfeatures = []
for column in data:
    Xfeatures.append(str(column))

Xfeatures.remove("fep_mh_willing")
Xfeatures.remove("cep_comf_cw")
Xfeatures.remove("cep_comf_sup")
Xfeatures.remove("mh_discuss_office")
print("Feature correlations")
#Calculating Cramer's V (gives a measure of association between 0 and 1)
corrs = {}
for feature in Xfeatures:
    confusion_matrix = pd.crosstab(data[feature], data['mh_discuss_office'])
    cm = confusion_matrix.to_numpy()
    X2 = ss.chi2_contingency(cm, correction=False)[0]
    n = np.sum(cm)
    minDim = min(cm.shape)-1
    V = np.sqrt((X2/n) / minDim)
    #print(str(feature) + ": " + str(V))
    #fdaklkjfkdsalfjklsthiskjkjkdd
    corrs[feature] = V
    
    
sort_orders = sorted(corrs.items(), key=lambda x: x[1], reverse=True)
print(sort_orders)


# In[ ]:




