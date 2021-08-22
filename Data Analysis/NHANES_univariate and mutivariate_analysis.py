
# coding: utf-8

# In[1]:


#UNIVARIATE ANALYSIS 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import numpy as np

da = pd.read_csv("nhanes_2015_2016.csv")
da.head()


# 1. Relabeling the marital status variable [DMDMARTL](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm#DMDMARTL) to have brief but informative character labels.  
# 2. Then constructing a frequency table of these values for all people, then for women only, and for men only.  
# 3. Then constructing these three frequency tables using only people whose age is between 30 and 40.

# In[2]:


da["DMDMARTL"] = da.DMDMARTL.replace({1: "Marr", 2: "Wid", 3: "Div", 4: "Sep", 5: "NMarr", 
                                       6: "Livtog",77: "Ref", 99: "Don't know"})
da.DMDMARTL.dropna()
da.RIAGENDR.dropna()
dam= da[da["RIAGENDR"]==1]
daf= da[da["RIAGENDR"]==2]
#print(da.DMDMARTL.value_counts()/da.DMDMARTL.value_counts().sum())
#print(daf.DMDMARTL.value_counts()/daf.DMDMARTL.value_counts().sum())
#print(dam.DMDMARTL.value_counts()/dam.DMDMARTL.value_counts().sum())

da["DMDMARTL"]



# In[3]:


da.RIDAGEYR.dropna()
dage = da[(da["RIDAGEYR"]>30) & (da["RIDAGEYR"]<40)]
dam= dage[da["RIAGENDR"]==1]
daf= dage[da["RIAGENDR"]==2]
print(dage.DMDMARTL.value_counts()/dage.DMDMARTL.value_counts().sum())
print(daf.DMDMARTL.value_counts()/daf.DMDMARTL.value_counts().sum())
print(dam.DMDMARTL.value_counts()/dam.DMDMARTL.value_counts().sum())


# 1. Restricting to the female population, stratify the subjects into age bands no wider than ten years, and construct the distribution of marital status within each age band.  
# 2. Within each age band, present the distribution in terms of proportions that must sum to 1.

# In[4]:


daf= da[da["RIAGENDR"]==2]
daf["agegrp"] = pd.cut(daf.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80])
daf['agegrp']
x=daf.groupby('agegrp')['DMDMARTL'].value_counts()
x=x/x.sum()
print(x)


# In[5]:


dam= da[da["RIAGENDR"]==1]
dam["agegrp"] = pd.cut(dam.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80])
y=dam.groupby('agegrp')['DMDMARTL'].value_counts()
y=y/y.sum()
print(y)


# 1. Constructing a histogram of the distribution of heights using the BMXHT variable in the NHANES sample.

# In[6]:


da.BMXHT.dropna(inplace=True)
da
import numpy as np
print(da['BMXHT'].isnull().values.any())
sns.distplot(da["BMXHT"], kde=False)


# In[7]:


da.BMXHT.dropna(inplace=True)
da
import numpy as np
print(da['BMXHT'].isnull().values.any())
plt.subplot(1,2,2)
sns.distplot(da["BMXHT"], kde=False, bins = 5)

#da.BMXHT.dropna(inplace=True)
#da
#import numpy as np
plt.subplot(1,2,1)
print(da['BMXHT'].isnull().values.any())
sns.distplot(da["BMXHT"], kde=False, bins = 30)


# In[8]:


dam["BMXHT"]


# In[9]:



daf= da[da["RIAGENDR"]==2]
dam= da[da["RIAGENDR"]==1]
dam.BMXHT.dropna(inplace=True)
daf.BMXHT.dropna(inplace=True)
da.RIAGENDR.dropna(inplace=True)

plt.subplot(2,1,1)
sns.distplot(daf["BMXHT"], kde=False)
plt.subplot(2,1,2)
sns.distplot(dam["BMXHT"], kde=False)

a= sns.FacetGrid(da, row= "RIAGENDR")
#a= a.map(plt.boxplot(dam["BMXHT"]))
#b= sns.FacetGrid(daf)
a= a.map(sns.boxplot, "BMXHT")


# Making a boxplot showing the distribution of within-subject differences between the first and second systolic blood pressure measurents ([BPXSY1](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BPX_I.htm#BPXSY1) and [BPXSY2](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BPX_I.htm#BPXSY2)).

# In[10]:


#plt.subplot(2,1,1)
sns.boxplot(x=da['BPXSY1']-da['BPXSY2'])
#plt.subplot(2,1,2)
#sns.boxplot(x=da['BPXSY2'])


# Identifying proportion of the subjects have a lower SBP on the second reading compared to the first?

# In[11]:


c=0
da['bpx']=da['BPXSY1'] + da['BPXSY2']
for i in da.T:
    if da.loc[i]["BPXSY1"]>da.loc[i]["BPXSY2"]:
        c=c+1
print(c/da.bpx.value_counts().sum())


# Making side-by-side boxplots of the two systolic blood pressure variables.

# In[12]:


plt.subplot(2,1,1)
sns.boxplot(x=da['BPXSY1'])
plt.subplot(2,1,2)
sns.boxplot(x=da['BPXSY2'])


# Constructing a frequency table of household sizes for people within each educational attainment category (the relevant variable is [DMDEDUC2](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm#DMDEDUC2)).  Convert the frequencies to proportions.

# In[13]:


print(da.columns)
x= da.groupby('DMDEDUC2')['DMDHHSIZ'].value_counts()
x= x/x.sum()
print(x)


# Restricting the sample to people between 30 and 40 years of age.  Then calculate the median household size for women and men within each level of educational attainment.

# In[14]:


dage = da[(da["RIDAGEYR"]>30) & (da["RIDAGEYR"]<40)]
dage.groupby('RIAGENDR')['DMDHHSIZ'].median()


# The participants can be clustered into "maked variance units" (MVU) based on every combination of the variables [SDMVSTRA](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm#SDMVSTRA) and [SDMVPSU](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm#SDMVPSU).  
# 
# I am calculating the mean age ([RIDAGEYR](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm#RIDAGEYR)), height ([BMXHT](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BMX_I.htm#BMXHT)), and BMI ([BMXBMI](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BMX_I.htm#BMXBMI)) for each gender ([RIAGENDR](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm#RIAGENDR)), within each MVU, and reporting the ratio between the largest and smallest mean (e.g. for height) across the MVUs.

# In[15]:


da.loc[:,['SDMVSTRA','SDMVPSU']] 
dam= da[da["RIAGENDR"]==1]
daf= da[da["RIAGENDR"]==2]
mmax= dam.groupby(['SDMVSTRA','SDMVPSU'])['RIDAGEYR'].mean().max()
mmin= dam.groupby(['SDMVSTRA','SDMVPSU'])['RIDAGEYR'].mean().min()
fmax= daf.groupby(['SDMVSTRA','SDMVPSU'])['RIDAGEYR'].mean().max()
fmin= daf.groupby(['SDMVSTRA','SDMVPSU'])['RIDAGEYR'].mean().min()
print(mmax,mmin,mmax/mmin)
print(fmax,fmin,fmax/fmin)


# Calculating the inter-quartile range (IQR) for age, height, and BMI for each gender and each MVU.  Report the ratio between the largest and smalles IQR across the MVUs.

# In[17]:


dam= da[da["RIAGENDR"]==1]
m25= dam.groupby(['SDMVSTRA','SDMVPSU'])['RIDAGEYR'].quantile(0.25).reset_index()
m75= dam.groupby(['SDMVSTRA','SDMVPSU'])['RIDAGEYR'].quantile(0.75).reset_index()
merged_df=m25.merge(m75, left_on=['SDMVSTRA','SDMVPSU'], right_on=['SDMVSTRA','SDMVPSU'],suffixes=('_25', '_75'))
merged_df['iqr']= merged_df['RIDAGEYR_75']-merged_df['RIDAGEYR_25']
print(merged_df)


# In[ ]:


#Multivariate analysis 


# 1. Making a scatterplot showing the relationship between the first and second measurements of diastolic blood pressure (BPXDI1 and BPXDI2). 
# 2. Also obtaining the 4x4 matrix of correlation coefficients among the first two systolic and the first two diastolic blood pressure measures.

# In[18]:


print(da.loc[:,['BPXSY1', 'BPXSY2','BPXDI1', 'BPXDI2']].dropna().corr())
#print(da.loc[:,['BPXDI1', 'BPXDI2']].dropna().corr())
sns.regplot(x='BPXDI2', y= 'BPXDI1', data= da, fit_reg= False, scatter_kws = {"alpha":0.4})


# The corelation between repeated diastolic blood pressures is lower than the repeated measurements of systolic blood presuure

# Constructing a grid of scatterplots between the first systolic and the first diastolic blood pressure measurement.  Stratify the plots by gender (rows) and by race/ethnicity groups (columns).

# In[20]:


sns.FacetGrid(da, row='RIAGENDR', col= 'RIDRETH1').map(plt.scatter,'BPXDI1','BPXSY1', alpha=0.5 ).add_legend()


# Corelation is the strongest for the last ethnic group. In general, the corelation is quite similar for both the genders in each ethnic group. 

# Using "violin plots" to compare the distributions of ages within groups defined by gender and educational attainment.

# In[ ]:


plt.figure(figsize=(20,4))
sns.violinplot(da.DMDEDUC2, da.RIDAGEYR, hue=da.RIAGENDR)
plt.figure(figsize=(20,4))
sns.boxplot(da.DMDEDUC2, da.RIDAGEYR, hue=da.RIAGENDR)


# In the first edu level, female age distribution is more left-skewed compared to male. The second and thid edu level age distributions are almost similar to a bell curve. Fourth is slightly right skewed, fifth female is right skewed, whereas male looks more of a normal distribution

# Using violin plots to compare the distributions of BMI within a series of 10-year age bands.  Also stratify these plots by gender.

# In[ ]:


plt.figure(figsize=(20,4))
da["agegrp"] = pd.cut(da.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80])
da['agegrp']
sns.violinplot(da.agegrp, da.BMXBMI, hue=da.RIAGENDR )


# Almost all the BMIs are right skewed except for males in the age group 40, 50 and 50, 60. Range of BMI for women is much smaller, and most of the BMI is distributed between 20-30.

# 1. Constructing a frequency table for the joint distribution of ethnicity groups ([RIDRETH1](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm#RIDRETH1)) and health-insurance status ([HIQ210](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/HIQ_I.htm#HIQ210)).  
# 2. Normalize the results so that the values within each ethnic group are proportions that sum to 1.

# In[21]:


da.groupby(['RIDRETH1', 'HIQ210']).size().unstack().fillna(0).apply(lambda z:z/z.sum(), axis=0)

