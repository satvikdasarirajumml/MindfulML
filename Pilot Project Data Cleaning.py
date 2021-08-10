#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Citations:
#Data Source: https://www.kaggle.com/osmi/mental-health-in-tech-2016
#Guidance for data cleaning from previous kaggle projects


# In[2]:


#Importing necessary packages 
import pandas as pd #pandas for dataframe manipulation
import numpy as np #numpy for basic calculations and array manipulation
import scipy as sp #scipy for toosl for scientific values
import seaborn as sns
import matplotlib.pyplot as plt #matplotlib for visualizations
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Reading csv file stored on computer as a dataframe (df)
df = pd.read_csv("C:\\Users\\satvi\\Downloads\\MentalHealthinTech.csv")
df.head()


# In[4]:


#Removing survey respondents that are self-employed (because they're not important in this study about tech workplaces)
df_a = df.copy()
df_a = df_a[df_a["Are you self-employed?"] == 0]


# In[5]:


#Some columns are completely empty (all nans or zero variance predictors)
#Removing these columns since they're just noise that obscures signal and hinders prediction
#List of columns to be removed
columns_to_be_removed = ["Do you know local or online resources to seek help for a mental health disorder?",
                         "If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?",
                         "If you have revealed a mental health issue to a client or business contact, do you believe this has impacted you negatively?",
                         "If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to coworkers or employees?",
                         "If you have revealed a mental health issue to a coworker or employee, do you believe this has impacted you negatively?",
                         "Do you believe your productivity is ever affected by a mental health issue?",
                         "If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?",
                         "Do you have medical coverage (private insurance or state-provided) which includes treatment of mental health issues?"]


#Removing irrelevant columns as well (columns that aren't going to tell us anything for prediction)
#List of irrelevant columns
irrelevant_columns = ["Are you self-employed?",
            "What US state or territory do you work in?",
           "What US state or territory do you live in?",
           "What country do you live in?",
           "Why or why not?",
           "Why or why not?.1"]

#Adding the list of irrelevant columns to list of columns to be removed
columns_to_be_removed.extend(irrelevant_columns)

#Removing/dropping the columns
df_b = df_a.copy()
df_b = df_b.drop(columns_to_be_removed,axis=1)


# In[6]:


#Now replacing verbal responses with number placeholders
df4 = df_b.copy()

rp_col = "How many employees does your company or organization have?"
rp_dt = {'1-5':1,
         '5-Jan':1, #1-5 was interpreted as 5th of January
        '6-25':6,
         '25-Jun':6, #6-25 was interpreted as 25th of June
        '26-100':26,
        '100-500':101,
        '500-1000':501,
        'More than 1000':1001}

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Is your primary role within your company related to tech/IT?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA


rp_col = "Does your employer provide mental health benefits as part of healthcare coverage?"
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        "I don't know":2, # responses in increasing negativity will be 2 onwards
        'No':3,
        'Not eligible for coverage / N/A':-1
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you know the options for mental health care available under your employer-provided coverage?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'I am not sure':2, # responses in increasing negativity will be 2 onwards
        'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)
rp_col = "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        "I don't know":2, # responses in increasing negativity will be 2 onwards
        'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Does your employer offer resources to learn more about mental health concerns and options for seeking help?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        "I don't know":2, # responses in increasing negativity will be 2 onwards
        'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA

rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        "I don't know":2, # responses in increasing negativity will be 2 onwards
        'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {"Very easy":1, # positive/yes response to qn will be 1
        "Somewhat easy":2, # responses in increasing negativity will be 2 onwards
        "Neither easy nor difficult":3,
         "I don't know":3,
         "Somewhat difficult":4,
         "Very difficult":5
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you think that discussing a mental health disorder with your employer would have negative consequences?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA


rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'Maybe':2, # responses in increasing negativity will be 2 onwards
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you think that discussing a physical health issue with your employer would have negative consequences?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'Maybe':2, # responses in increasing negativity will be 2 onwards,
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Would you feel comfortable discussing a mental health disorder with your coworkers?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'Maybe':2, # responses in increasing negativity will be 2 onwards,
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

rp_col = "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'Maybe':2, # responses in increasing negativity will be 2 onwards,
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you feel that your employer takes mental health as seriously as physical health?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        "I don't know":2, # responses in increasing negativity will be 2 onwards,
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'No':2, # responses in increasing negativity will be 2 onwards,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you have previous employers?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {0:2 # replace 0 (no) with 2 for consistency
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Have your previous employers provided mental health benefits?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, they all did':1, # positive/yes response to qn will be 1
        'Some did':2, # responses in increasing negativity will be 2 onwards,
        "I don't know":3,
         'No, none did':4
        }
df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Were you aware of the options for mental health care provided by your previous employers?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, I was aware of all of them':1, # positive/yes response to qn will be 1
        'I was aware of some':2, # responses in increasing negativity will be 2 onwards,
        'No, I only became aware later':3,
         'N/A (not currently aware)':4
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, they all did':1, # positive/yes response to qn will be 1
        'Some did':2, # responses in increasing negativity will be 2 onwards,
        "I don't know":3,
         'None did':4
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Did your previous employers provide resources to learn more about mental health issues and how to seek help?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, they all did':1, # positive/yes response to qn will be 1
        'Some did':2, # responses in increasing negativity will be 2 onwards,
        "I don't know":3,
         'None did':4
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, always':1, # positive/yes response to qn will be 1
        'Sometimes':2, # responses in increasing negativity will be 2 onwards,
        "I don't know":3,
         'No':4
        }


df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you think that discussing a mental health disorder with previous employers would have negative consequences?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, all of them':1, # positive/yes response to qn will be 1
        'Some of them':2, # responses in increasing negativity will be 2 onwards,
        "I don't know":3,
         'None of them':4
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you think that discussing a physical health issue with previous employers would have negative consequences?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, all of them':1, # positive/yes response to qn will be 1
        'Some of them':2, # responses in increasing negativity will be 2 onwards,
         'None of them':3
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Would you have been willing to discuss a mental health issue with your previous co-workers?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, at all of my previous employers':1, # positive/yes response to qn will be 1
        'Some of my previous employers':2, # responses in increasing negativity will be 2 onwards,
         'No, at none of my previous employers':3
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Would you have been willing to discuss a mental health issue with your direct supervisor(s)?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, at all of my previous employers':1, # positive/yes response to qn will be 1
        'Some of my previous employers':2, # responses in increasing negativity will be 2 onwards,
         "I don't know":3,
         'No, at none of my previous employers':4
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Did you feel that your previous employers took mental health as seriously as physical health?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, they all did':1, # positive/yes response to qn will be 1
        'Some did':2, # responses in increasing negativity will be 2 onwards,
        "I don't know":3,
         'None did':4
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, all of them':1, # positive/yes response to qn will be 1
        'Some of them':2, # responses in increasing negativity will be 2 onwards,
         'None of them':3
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Would you be willing to bring up a physical health issue with a potential employer in an interview?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'Maybe':2, # responses in increasing negativity will be 2 onwards,
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Would you bring up a mental health issue with a potential employer in an interview?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'Maybe':2, # responses in increasing negativity will be 2 onwards,
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you feel that being identified as a person with a mental health issue would hurt your career?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, it has':1, # positive/yes response to qn will be 1
        'Yes, I think it would':2, # responses in increasing negativity will be 2 onwards,
        'Maybe':3,
         "No, I don't think it would":4,
         'No, it has not':5
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, they do':1, # positive/yes response to qn will be 1
         'Yes, I think they would':2, # responses in increasing negativity will be 2 onwards,
        'Maybe':3,
         "No, I don't think they would":4,
         'No, they do not':5
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "How willing would you be to share with friends and family that you have a mental illness?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Very open':1, # positive/yes response to qn will be 1
         'Somewhat open':2, # responses in increasing negativity will be 2 onwards,
        'Neutral':3,
         'Somewhat not open':4,
         'Not open at all':5,
         'Not applicable to me (I do not have a mental illness)':-1
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes, I experienced':1, # positive/yes response to qn will be 1
         'Yes, I observed':2, # responses in increasing negativity will be 2 onwards,
        'Maybe/Not sure':3,
         'No':4,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?"
# nan values is 55.41%; unsure what is the cause of nan values
# drop column
df4 = df4.drop([rp_col],axis=1)

###
rp_col = "Do you have a family history of mental illness?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        "I don't know":2, # responses in increasing negativity will be 2 onwards
        'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Have you had a mental health disorder in the past?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'Maybe':2, # responses in increasing negativity will be 2 onwards,
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

rp_col = "Do you currently have a mental health disorder?"
# potential target column or key X column
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'Maybe':2, # responses in increasing negativity will be 2 onwards,
         'No':3,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Have you been diagnosed with a mental health condition by a medical professional?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Yes':1, # positive/yes response to qn will be 1
        'No':2, # responses in increasing negativity will be 2 onwards,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "Have you ever sought treatment for a mental health issue from a mental health professional?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {1:1, # positive/yes response to qn will be 1
        0:2, # responses in increasing negativity will be 2 onwards,
        }

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Often':1, # positive/yes response to qn will be 1
        'Sometimes':2, # responses in increasing negativity will be 2 onwards,
        'Rarely':3,
        'Never':4,
        'Not applicable to me':-1}

df4[rp_col] = df4[rp_col].replace(rp_dt)

###
rp_col = "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Often':1, # positive/yes response to qn will be 1
        'Sometimes':2, # responses in increasing negativity will be 2 onwards,
        'Rarely':3,
        'Never':4,
        'Not applicable to me':-1}

df4[rp_col] = df4[rp_col].replace(rp_dt)

###

rp_col = "Do you work remotely?"
df4[rp_col] = df4[rp_col].fillna(-1) #for NA
rp_dt = {'Always':1, # positive/yes response to qn will be 1
        'Sometimes':2, # responses in increasing negativity will be 2 onwards,
        'Never':3,
       }

df4[rp_col] = df4[rp_col].replace(rp_dt)

#####
df4.describe(include='all')


# In[7]:


#Remove respondents who don't work in a tech role or a nontech role in a tech company
df5 = df4.copy()
df5 = df5[df5["Is your primary role within your company related to tech/IT?"].isin([-1,1])]


# In[8]:


#Renaming the column after the removal of respondents who didn't work either in a tech role or a nontech role
df5 = df5.rename(columns={"Is your employer primarily a tech company/organization?":"tech_company_or_role"})


# In[9]:


#Defining a replacement function to get rid of weird age values (e.g., 323)
def rp_age(age):
    '''
    Replaces age below min or age above max with mode age.
    Else, returns age.'''
    mode = 32
    low,up = 13,72
    if age < 13: return mode
    elif age > 72: return mode
    else: return int(age)

df6 = df5.copy()

#Getting rid of mental health disorder diagnosis for now (will come back to this)
df6 = df6.drop(["If yes, what condition(s) have you been diagnosed with?",
         "If maybe, what condition(s) do you believe you have?",
               "If so, what condition(s) were you diagnosed with?"],axis=1)

df6['What is your age?'] = df6['What is your age?'].apply(rp_age)


# In[10]:


#Standardizing and ennumerating gender verbal responses
df7 = df6.copy()
# prepare replacement lists
male_ls = ['Male','male', 'Male ', 'M', 'm', 'man', 'Cis male',
           'Male.', 'Male (cis)', 'Man', 'Sex is male',
           'cis male', 'Malr', 'Dude', "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ",
           'mail', 'M|', 'male ', 'Cis Male', 'Male (trans, FtM)',
           'cisdude', 'cis man', 'MALE']
# FYI: cisgender: describes a person who identifies as the same gender assigned at birth
female_ls = ['Female','female', 'I identify as female.', 'female ',
             'Female assigned at birth ', 'F', 'Woman', 'fm', 'f',
             'Cis female', 'Transitioned, M2F', 'Female or Multi-Gender Femme',
             'Female ', 'woman', 'female/woman', 'Cisgender Female', 
             'mtf', 'fem', 'Female (props for making this a freeform field, though)',
             ' Female', 'Cis-woman', 'AFAB', 'Transgender woman',
             'Cis female ']

# FYI: AFAB: assigned female at birth
other_ls = ['Bigender', 'non-binary,', 'Genderfluid (born female)',
            'Other/Transfeminine', 'Androgynous', 'male 9:1 female, roughly',
            'nb masculine', 'genderqueer', 'Human', 'Genderfluid',
            'Enby', 'genderqueer woman', 'Queer', 'Agender', 'Fluid',
            'Genderflux demi-girl', 'female-bodied; no feelings about gender',
            'non-binary', 'Male/genderqueer', 'Nonbinary', 'Other', 'none of your business',
            'Unicorn', 'human', 'Genderqueer']

# replace gender values with numberic labels
df7["What is your gender?"] = df7["What is your gender?"].replace(male_ls,1)
df7["What is your gender?"] = df7["What is your gender?"].replace(female_ls,2)
df7["What is your gender?"] = df7["What is your gender?"].replace(other_ls,3)
df7["What is your gender?"] = df7["What is your gender?"].fillna(3)


# In[11]:


#Ennumerating country name automatically
df8 = df7.copy()
country_rp_dt = {}
for idx, name in enumerate(df8['What country do you work in?'].unique()):
#     print(idx, name)
    country_rp_dt[name] = idx
# country_rp_dt
df8['What country do you work in?'] = df8['What country do you work in?'].replace(country_rp_dt)


# In[12]:


#Getting rid of the work position column
df8 = df8.rename(columns={"Is your employer primarily a tech company/organization?":"tech_company_or_role"})
df8 = df8.drop(["Which of the following best describes your work position?"],axis=1)
df8 = df8.drop(["Is your primary role within your company related to tech/IT?"],axis=1)


# In[13]:


#Adding simpler column titles
df10 = df8.copy()

df_rn_dt = {
    "How many employees does your company or organization have?":"num_employees",
    "Does your employer provide mental health benefits as part of healthcare coverage?":"cep_benefits",
    "Do you know the options for mental health care available under your employer-provided coverage?":"cep_know_options",
    "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?":"cep_discuss",
    "Does your employer offer resources to learn more about mental health concerns and options for seeking help?":"cep_learn",
    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?":"cep_anon",
    "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:":"cep_mh_leave",
    "Do you think that discussing a mental health disorder with your employer would have negative consequences?":"cep_mh_ncsq",
    "Do you think that discussing a physical health issue with your employer would have negative consequences?":"cep_ph_ncsq",
    "Would you feel comfortable discussing a mental health disorder with your coworkers?":"cep_comf_cw",
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?":"cep_comf_sup",
    "Do you feel that your employer takes mental health as seriously as physical health?":"cep_serious",
    "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?":"cep_others_ncsq",
    "Do you have previous employers?":"pep_have",
    "Have your previous employers provided mental health benefits?":"pep_benefits",
    "Were you aware of the options for mental health care provided by your previous employers?":"pep_know_options",
    "Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?":"pep_discuss",
    "Did your previous employers provide resources to learn more about mental health issues and how to seek help?":"pep_learn",
    "Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?":"pep_anon",
    "Do you think that discussing a mental health disorder with previous employers would have negative consequences?":"pep_mh_ncsq",
    "Do you think that discussing a physical health issue with previous employers would have negative consequences?":"pep_ph_ncsq",
    "Would you have been willing to discuss a mental health issue with your previous co-workers?":"pep_comf_cw",
    "Would you have been willing to discuss a mental health issue with your direct supervisor(s)?":"pep_comf_sup",
    "Did you feel that your previous employers took mental health as seriously as physical health?":"pep_serious",
    "Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?":"pep_others_ncsq",
    "Would you be willing to bring up a physical health issue with a potential employer in an interview?":"fep_ph_willing",
    "Would you bring up a mental health issue with a potential employer in an interview?":"fep_mh_willing",
    "Do you feel that being identified as a person with a mental health issue would hurt your career?":"hurt_career",
    "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?":"cw_view_neg",
    "How willing would you be to share with friends and family that you have a mental illness?":"comf_ff",
    "Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?":"neg_response",
    "Do you have a family history of mental illness?":"mh_fam_hist",
    "Have you had a mental health disorder in the past?":"mh_hist",
    "Do you currently have a mental health disorder?":"mh_cur",
    "Have you been diagnosed with a mental health condition by a medical professional?":"mh_diag_pro",
    "Have you ever sought treatment for a mental health issue from a mental health professional?":"sought_treat",
    "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?":"work_affect_effect",
    "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?":"work_affect_ineffect",
    "What is your age?":"age",
    "What is your gender?":"gender",
    "What country do you work in?":"work_country",
    "Do you work remotely?":"work_remote"
}

df10=df10.rename(columns=df_rn_dt)


# In[14]:


df10.head()


# In[15]:


#Saving cleaned data onto computer
df10.to_csv("C:\\Users\\satvi\\Downloads\\CleanedMentalHealthinTech.csv", index = False)

