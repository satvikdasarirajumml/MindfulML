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


#setting RSEED allows for reproducible runs, otherwise each run will be random and have slightly different results
RSEED = 1


# In[3]:


df = pd.read_csv("C:\\Users\\satvi\\Downloads\\CleanedMentalHealthinTech.csv")


# In[4]:


#Classification task 1: predict if one is willing to raise mental health issues to their employer
from sklearn.model_selection import train_test_split

x_col = ['num_employees',
 'tech_company_or_role',
 'comf_ff',
 'mh_fam_hist',
 'mh_hist',
 'mh_cur',
 'age',
 'gender',
 'work_country',
 'work_remote',
 'cep_benefits',
 'cep_know_options',
 'cep_discuss',
 'cep_learn',
 'cep_anon',
 'cep_mh_leave',
 'cep_mh_ncsq',
 'cep_ph_ncsq','cep_comf_cw',
 'cep_comf_sup',
 'cep_serious',
 'cep_others_ncsq',
 'pep_have',
 'pep_benefits',
 'pep_know_options',
 'pep_discuss',
 'pep_learn',
 'pep_anon',
 'pep_mh_ncsq',
 'pep_ph_ncsq',
 'pep_comf_cw',
 'pep_comf_sup',
 'pep_serious',
 'pep_others_ncsq',
 'hurt_career',
 'cw_view_neg',
 'neg_response',
 'work_affect_effect',
 'work_affect_ineffect']

X = df[x_col]
y1 = df["fep_mh_willing"]


# 30% examples in test data
X_train, X_test, y_train, y_test = train_test_split(X, y1, 
                                                    test_size = 0.25,
                                                   random_state=90)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(bootstrap=True, class_weight='balanced', criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
            oob_score=False, random_state=1, verbose=0, warm_start=False)

#Fits on training data
model.fit(X_train, y_train)


# In[5]:


#testing model on training data (data it has already seen)
train_rf_predictions = model.predict(X_train)
train_rf_probs = model.predict_proba(X_train)[:, 1]

#testing model on testing data (new data)
rf_predictions = model.predict(X_test)
rf_probs = model.predict_proba(X_test)[:, 1]


# In[6]:


#importing the necessary performance metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score

print('Random Forest Training Accuracy Score: ' + str(accuracy_score(y_train, train_rf_predictions)))
print('Random Forest Testing Accuracy Score: ' + str(accuracy_score(y_test, rf_predictions)))


# In[7]:


features = list(X_train.columns)
fi_model = pd.DataFrame({'feature': features,
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)
 
fi_model.head(10)


# In[8]:


#Classification task 2: predict if one will seek treatment

x_col = ['num_employees',
 'tech_company_or_role',
 'comf_ff',
 'mh_fam_hist',
 'mh_hist',
 'mh_cur',
 'age',
 'gender',
 'work_country',
 'work_remote',
 'cep_benefits',
 'cep_know_options',
 'cep_discuss',
 'cep_learn',
 'cep_anon',
 'cep_mh_leave',
 'cep_mh_ncsq',
 'cep_ph_ncsq','cep_comf_cw',
 'cep_comf_sup',
 'cep_serious',
 'cep_others_ncsq',
 'pep_have',
 'pep_benefits',
 'pep_know_options',
 'pep_discuss',
 'pep_learn',
 'pep_anon',
 'pep_mh_ncsq',
 'pep_ph_ncsq',
 'pep_comf_cw',
 'pep_comf_sup',
 'pep_serious',
 'pep_others_ncsq',
 'hurt_career',
 'cw_view_neg',
 'neg_response',
 'work_affect_effect',
 'work_affect_ineffect']

X = df[x_col]
X = X.drop(["mh_hist","mh_cur","work_affect_effect", "work_affect_ineffect"],axis=1)

y2 = df["sought_treat"]


# 30% examples in test data
X_train, X_test, y_train, y_test = train_test_split(X, y2, 
                                                    test_size = 0.25,
                                                   random_state=90)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(bootstrap=True, class_weight='balanced', criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
            oob_score=False, random_state=1, verbose=0, warm_start=False)

#Fits on training data
model.fit(X_train, y_train)



# In[9]:


#testing model on training data (data it has already seen)
train_rf_predictions = model.predict(X_train)
train_rf_probs = model.predict_proba(X_train)[:, 1]

#testing model on testing data (new data)
rf_predictions = model.predict(X_test)
rf_probs = model.predict_proba(X_test)[:, 1]


# In[10]:


print('Random Forest Training Accuracy Score: ' + str(accuracy_score(y_train, train_rf_predictions)))
print('Random Forest Testing Accuracy Score: ' + str(accuracy_score(y_test, rf_predictions)))


# In[11]:


features = list(X_train.columns)
fi_model = pd.DataFrame({'feature': features,
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)
 
fi_model.head(10)


# In[12]:


#Introducing a constructed feature (mh_discuss_office): is an individual willing to discuss mhd in the office place?
data = pd.read_csv("C:\\Users\\satvi\\Downloads\\CleanedMentalHealthinTech.csv")

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


# In[13]:


#Classification task 2: predict if one will seek treatment

x_col = ['num_employees',
 'tech_company_or_role',
 'comf_ff',
 'mh_fam_hist',
 'mh_hist',
 'mh_cur',
 'age',
 'gender',
 'work_country',
 'work_remote',
 'cep_benefits',
 'cep_know_options',
 'cep_discuss',
 'cep_learn',
 'cep_anon',
 'cep_mh_leave',
 'cep_mh_ncsq',
 'cep_ph_ncsq','cep_comf_cw',
 'cep_comf_sup',
 'cep_serious',
 'cep_others_ncsq',
 'pep_have',
 'pep_benefits',
 'pep_know_options',
 'pep_discuss',
 'pep_learn',
 'pep_anon',
 'pep_mh_ncsq',
 'pep_ph_ncsq',
 'pep_comf_cw',
 'pep_comf_sup',
 'pep_serious',
 'pep_others_ncsq',
 'hurt_career',
 'cw_view_neg',
 'neg_response',
 'work_affect_effect',
 'work_affect_ineffect']

X = data[x_col]
X = X.drop(["pep_comf_cw", "pep_comf_sup"],axis=1)

y3 = data["mh_discuss_office"]


# 30% examples in test data
X_train, X_test, y_train, y_test = train_test_split(X, y3, 
                                                    test_size = 0.25,
                                                   random_state=90)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(bootstrap=True, class_weight='balanced', criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
            oob_score=False, random_state=1, verbose=0, warm_start=False)

#Fits on training data
model.fit(X_train, y_train)



# In[14]:


#testing model on training data (data it has already seen)
train_rf_predictions = model.predict(X_train)
train_rf_probs = model.predict_proba(X_train)[:, 1]

#testing model on testing data (new data)
rf_predictions = model.predict(X_test)
rf_probs = model.predict_proba(X_test)[:, 1]


# In[15]:


print('Random Forest Training Accuracy Score: ' + str(accuracy_score(y_train, train_rf_predictions)))
print('Random Forest Testing Accuracy Score: ' + str(accuracy_score(y_test, rf_predictions)))


# In[16]:


features = list(X_train.columns)
fi_model = pd.DataFrame({'feature': features,
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)
 
fi_model.head(10)


# In[ ]:




