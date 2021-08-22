
# coding: utf-8

# # Hypothesis tests using NHANES data

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import numpy as np
import scipy.stats.distributions as dist

da = pd.read_csv("nhanes_2015_2016.csv")


# Conducting a hypothesis test (at the 0.05 level) for the null hypothesis that the proportion of women who smoke is equal to the proportion of men who smoke.

# In[2]:


da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})
da["SMQ020x"] = da.SMQ020.replace({1: "Yes", 2: "No", 7: np.nan, 9: np.nan})  


# In[3]:


dx = da[["SMQ020x", "RIDAGEYR", "RIAGENDRx"]].dropna()


# In[4]:


# Summarizing the data by caclculating the proportion of yes responses and the sample size
p = dx.groupby("RIAGENDRx")["SMQ020x"].agg([lambda z: np.mean(z=="Yes"), "size"])
p.columns = ["Smoke", "N"]
print(p)

# The pooled rate of yes responses, and the standard error of the estimated difference of proportions
p_comb = (dx.SMQ020x == "Yes").mean()
va = p_comb * (1 - p_comb)
se = np.sqrt(va * (1 / p.N.Female + 1 / p.N.Male))

# Calculating the test statistic and its p-value
test_stat = (p.Smoke.Female - p.Smoke.Male) / se
pvalue = 2*dist.norm.cdf(-np.abs(test_stat))
print(test_stat, pvalue)


# In[5]:


dx_females = dx.loc[dx.RIAGENDRx=="Female", "SMQ020x"].replace({"Yes": 1, "No": 0})
dx_males = dx.loc[dx.RIAGENDRx=="Male", "SMQ020x"].replace({"Yes": 1, "No": 0})
sm.stats.ttest_ind(dx_females, dx_males)


# Constructing three 95% confidence intervals: one for the proportion of women who smoke, one for the proportion of men who smoke, and one for the difference in the rates of smoking between women and men.

# In[6]:


pd.crosstab(dx.SMQ020x, dx.RIAGENDRx)


# In[7]:


dz = dx.groupby(dx.RIAGENDRx).agg({"SMQ020x": [lambda x: np.mean(x=="Yes"), np.size]})
dz.columns = ["Proportion", "Total_n"] # The default column names are unclear, so we replace them here
dz


# In[8]:


sm.stats.proportion_confint(906, 906+2066)  


# In[9]:


sm.stats.proportion_confint(1413, 1413+1340)


# In[10]:


p = dz.Proportion.Female # Female proportion
n = dz.Total_n.Female # Total number of females
se_female = np.sqrt(p * (1 - p) / n)
print(se_female)

p = dz.Proportion.Male # Male proportion
n = dz["Total_n"].Male # Total number of males
se_male = np.sqrt(p * (1 - p) / n)
print(se_male)


# In[11]:


se_diff = np.sqrt(se_female**2 + se_male**2)
se_diff


# In[12]:


d = dz.Proportion.Female - dz.Proportion.Male
lcb = d - 2*se_diff
ucb = d + 2*se_diff
print(lcb, ucb)


# 1. Partitioning the population into two groups based on whether a person has graduated college or not, using the educational attainment variable [DMDEDUC2](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm#DMDEDUC2).  
# 2. Then conducting a test of the null hypothesis that the average heights (in centimeters) of the two groups are equal. 
# 3. Next, converting the heights from centimeters to inches, and conducting a test of the null hypothesis that the average heights (in inches) of the two groups are equal.

# In[13]:


da.columns


# In[14]:


da["DMDEDUC2x"] = da.DMDEDUC2.replace({1: "<9", 2: "9-11", 3: "HS/GED", 4: "Some college/AA", 5: "College", 
                                      7: "Refused", 9: "Don't know"})


# In[15]:


d_educated = da.loc[(da["DMDEDUC2x"] == "Some college/AA") | (da["DMDEDUC2x"] == "College"), :]
d_educated = d_educated[["DMDEDUC2x", 'BMXHT']].dropna(inplace = False)

d_notedu = da.loc[(da["DMDEDUC2x"] != "Some college/AA") | (da["DMDEDUC2x"] != "College"), :]
d_notedu = d_notedu[["DMDEDUC2x",'BMXHT']].dropna(inplace = False)


# In[16]:


sm.stats.ttest_ind(d_educated['BMXHT'], d_notedu['BMXHT'])


# In[17]:


da["BMXHT_in"] = da.BMXHT/2.54


# In[18]:


d_educated_in = da.loc[(da["DMDEDUC2x"] == "Some college/AA") | (da["DMDEDUC2x"] == "College"), :]
d_educated_in = d_educated_in[["DMDEDUC2x", 'BMXHT_in']].dropna(inplace = False)

d_notedu_in = da.loc[(da["DMDEDUC2x"] != "Some college/AA") | (da["DMDEDUC2x"] != "College"), :]
d_notedu_in = d_notedu_in[["DMDEDUC2x",'BMXHT_in']].dropna(inplace = False)


# In[19]:


sm.stats.ttest_ind(d_educated_in["BMXHT_in"], d_notedu_in["BMXHT_in"])


# Conducting a hypothesis test of the null hypothesis that the average BMI for men between 30 and 40 is equal to the average BMI for men between 50 and 60.  Then carry out this test again after log transforming the BMI values.

# In[20]:


d_3 = da[['BMXBMI', "RIAGENDRx", "RIDAGEYR"]].dropna()
d_31 = d_3.loc[(d_3["RIAGENDRx"] == "Male") & (d_3.RIDAGEYR >= 30) & (d_3.RIDAGEYR <= 40), :]
d_32 = d_3.loc[(d_3["RIAGENDRx"] == "Male") & (d_3.RIDAGEYR >= 50) & (d_3.RIDAGEYR <= 60), :]
print(sm.stats.ttest_ind(d_31['BMXBMI'], d_32['BMXBMI']))


# Suppose we wish to compare the mean BMI between college graduates and people who have not graduated from college, focusing on women between the ages of 30 and 40.  First, I am considering the variance of BMI within each of these subpopulations using graphical techniques, and through the estimated subpopulation variances.  Then, I'm calculate pooled and unpooled estimates of the standard error for the difference between the mean BMI in the two populations being compared.  Finally, I'm test the null hypothesis that the two population means are equal, using each of the two different standard errors.

# In[21]:


d_educated_5 = da.loc[(da["DMDEDUC2x"] == "Some college/AA") | (da["DMDEDUC2x"] == "College") & (da["RIAGENDRx"] == "Female") & (d_3.RIDAGEYR >= 30) & (d_3.RIDAGEYR <= 40), 'BMXBMI'].dropna()
#d_educated_5 = d_educated_5[["DMDEDUC2x", 'BMXBMI']].dropna(inplace = False)
d_educated_5 = sm.stats.DescrStatsW(d_educated_5)
d_notedu_5 = da.loc[(da["DMDEDUC2x"] != "Some college/AA") | (da["DMDEDUC2x"] != "College") & (da["RIAGENDRx"] == "Female") & (d_3.RIDAGEYR >= 30) & (d_3.RIDAGEYR <= 40), 'BMXBMI'].dropna()
#d_notedu_5 = d_notedu_5[["DMDEDUC2x",'BMXBMI']].dropna(inplace = False)
d_notedu_5 = sm.stats.DescrStatsW(d_notedu_5)


# In[22]:


sm.stats.CompareMeans(d_educated_5, d_notedu_5).ztest_ind(usevar='pooled')


# In[23]:


sm.stats.CompareMeans(d_educated_5, d_notedu_5).ztest_ind(usevar='unequal')


# Now I'm conducting a test of the null hypothesis that the first and second diastolic blood pressure measurements within a subject have the same mean values.

# In[24]:


dx_5 = da[["BPXDI1", "BPXDI2"]].dropna()
db = dx_5.BPXDI1 - dx_5.BPXDI2
sm.stats.ztest(db)


# Pretending that the first and second diastolic blood pressure measurements were taken on different people, I'vd modfied the analysis above as appropriate for this setting.

# In[25]:


d_dia_1 = sm.stats.DescrStatsW(dx_5.BPXDI1)
d_dia_2 = sm.stats.DescrStatsW(dx_5.BPXDI2)
sm.stats.CompareMeans(d_dia_1, d_dia_2).ztest_ind(usevar='pooled')


# In[26]:


sm.stats.CompareMeans(d_dia_1, d_dia_2).ztest_ind(usevar='unequal')


# # Confidence intervals using NHANES data

# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm

da = pd.read_csv("nhanes_2015_2016.csv")


# In[28]:


da.columns


# 1. Restricting the sample to women between 35 and 50, then use the marital status variable [DMDMARTL](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm#DMDMARTL) to partition this sample into two groups - women who are currently married, and women who are not currently married.  
# 2. Within each of these groups, I'm calculating the proportion of women who have completed college.  
# 3. I'm calculating 95% confidence intervals for each of these proportions.

# In[29]:


da["DMDEDUC2x"] = da.DMDEDUC2.replace({1: "<9", 2: "9-11", 3: "HS/GED", 4: "Some college/AA", 5: "College", 
                                      7: "Refused", 9: "Don't know"})


# In[30]:


da["DMDMARTL"] = da.DMDMARTL.replace({1: "Married", 2: "Not Married", 3: "Not Married", 4: "Not Married", 5: "Not Married", 6: "Not Married", 77: "Refused", 99: "Don't know"})


# In[31]:


da_married = da.loc[(da["DMDMARTL"]!= "Refused") & (da['RIDAGEYR'] >= 35) & (da['RIDAGEYR'] <= 50) & (da["RIAGENDR"] == 2), :]


# In[32]:


da_married


# In[33]:


da_married_new = da_married[["DMDMARTL", "DMDEDUC2x"]].dropna(inplace = False)


# In[34]:


pd.crosstab(da_married_new.DMDMARTL , da_married_new.DMDEDUC2x)


# In[35]:


da_prop = da_married_new.groupby(da_married_new.DMDMARTL).agg({"DMDEDUC2x": [lambda x: np.mean(x=="College"), np.size]})
da_prop.columns = ["Proportion", "Total_n"] # The default column names are unclear, so we replace them here
da_prop


# In[36]:


#calculating confidence intervals manually 
p = da_prop.Proportion.Married # Female proportion
n = da_prop.Total_n.Married # Total number of females
se_women = np.sqrt(p * (1 - p) / n)


lcb = p - 1.96 * np.sqrt(p * (1 - p) / n)  
ucb = p + 1.96 * np.sqrt(p * (1 - p) / n)  
print(lcb, ucb)


# In[37]:


p= da_prop.Proportion['Not Married'] # Female proportion
n = da_prop.Total_n['Not Married'] # Total number of females
se_women = np.sqrt(p * (1 - p) / n)

lcb_nm = p - 1.96 * np.sqrt(p * (1 - p) / n)  
ucb_nm = p + 1.96 * np.sqrt(p * (1 - p) / n)  
print(lcb_nm, ucb_nm)


# In[38]:


diff_m = ucb - lcb
diff_nm = ucb_nm - lcb_nm
print(diff_m, diff_nm)


# The proportion of married women between the ages 35 and 50 who studied till college is closer to 0.5. The Standard error is maximized as the proportion approaches 0.5. The higher the SE the more wider the confidence intervals. Hence the confidence intervals are wider for women who are married and have graduated college. 

# We can say with 95% confidence that in the population, the proportion of women who are married and have graduated from college within the ages 35 and 50 is definitely higher than women who are not married and have graduated college within the same age gap. 

# Next,
# 
# 1. Constructing a 95% confidence interval for the proportion of smokers who are female. 
# 2. Constructing a 95% confidence interval for the proportion of smokers who are male. Construct a 95% confidence interval for the **difference** between those two gender proportions.

# In[39]:


da["SMQ020x"] = da.SMQ020.replace({1: "Yes", 2: "No", 7: np.nan, 9: np.nan})  # np.nan represents a missing value
da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})


# In[40]:


dx = da[["SMQ020x", "RIAGENDRx"]].dropna()  # dropna drops cases where either variable is missing
pd.crosstab(dx.SMQ020x, dx.RIAGENDRx)


# In[41]:


dz = dx.groupby(dx.RIAGENDRx).agg({"SMQ020x": [lambda x: np.mean(x=="Yes"), np.size]})
dz.columns = ["Proportion", "Total_n"] # The default column names are unclear, so we replace them here
dz


# In[42]:


sm.stats.proportion_confint(906, 906+2066) 


# In[43]:


sm.stats.proportion_confint(1413, 1413+1340)


# In[44]:


p = dz.Proportion.Female # Female proportion
n = dz.Total_n.Female # Total number of females
se_female = np.sqrt(p * (1 - p) / n)
print(se_female)

p = dz.Proportion.Male # Male proportion
n = dz["Total_n"].Male # Total number of males
se_male = np.sqrt(p * (1 - p) / n)
print(se_male)


# In[45]:


se_diff = np.sqrt(se_female**2 + se_male**2)
se_diff


# In[46]:


d = dz.Proportion.Female - dz.Proportion.Male
lcb = d - 2*se_diff
ucb = d + 2*se_diff
print(lcb, ucb)


# In[47]:


#calculating smoking rates within different age bands 
# Calculate the smoking rates within age/gender groups
da["agegrp"] = pd.cut(da.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80])
pr = da.groupby(["agegrp", "RIAGENDRx"]).agg({"SMQ020x": lambda x: np.mean(x=="Yes")}).unstack()
pr.columns = ["Female", "Male"]

# The number of people for each calculated proportion
dn = da.groupby(["agegrp", "RIAGENDRx"]).agg({"SMQ020x": np.size}).unstack()
dn.columns = ["Female", "Male"]

# Standard errors for each proportion
se = np.sqrt(pr * (1 - pr) / dn)

# Standard error for the difference in female/male smoking rates in every age band
se_diff = np.sqrt(se.Female**2 + se.Male**2)

# Standard errors for the difference in smoking rates between genders, within age bands

# The difference in smoking rates between genders
pq = pr.Female - pr.Male

x = np.arange(pq.size)

pp = sns.pointplot(x, pq.values, color='black')
sns.pointplot(x, pq - 2*se_diff)
sns.pointplot(x, pq + 2*se_diff)
pp.set_xticklabels(pq.index)
pp.set_xlabel("Age group")
pp.set_ylabel("Female - male smoking proportion")
print(x)
print(pq.index)


# 1. The smoking is more predominant for males than females
# 2. The difference is more significant for older people.
# 3. Confidence bands are wider than the one for the entire age group because stratified samples have smaller size

# Now, 
# 
# 1. I'm constructing a 95% interval for height ([BMXHT](https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BMX_I.htm#BMXHT)) in centimeters.  Then convert height from centimeters to inches by dividing by 2.54, and I'm constructing a 95% confidence interval for height in inches.  
# 2. Finally, I'm converting the endpoints (the lower and upper confidence limits) of the confidence interval from inches to back to centimeters.  

# In[48]:


mean = np.mean(da.BMXHT)


# In[49]:


standev = np.std(da.BMXHT)


# In[50]:


total = np.size(da.BMXHT)


# In[51]:


stan_Error = standev/np.sqrt(total)


# In[52]:


lcb_cm = mean - 1.96 * stan_Error/ np.sqrt(total)
ucb_cm = mean + 1.96 * stan_Error/ np.sqrt(total)
print(lcb_cm, ucb_cm)


# In[53]:


da["BMXHT_in"] = da.BMXHT/2.54


# In[54]:


mean_in = np.mean(da["BMXHT_in"])


# In[55]:


standev_in = np.std(da["BMXHT_in"])


# In[56]:


total_in = np.size(da["BMXHT_in"])


# In[57]:


stan_Error_in = standev_in/np.sqrt(total)


# In[58]:


lcb_in = mean_in - 1.96 * stan_Error_in/ np.sqrt(total_in)
ucb_in = mean_in + 1.96 * stan_Error_in/ np.sqrt(total_in)
print(lcb_in, ucb_in)


# Confidence intervals for cm/2.54 is the confidence intervals for the inches

# 1. I'm partitioning the sample based on 10-year age bands, i.e. the resulting groups will consist of people with ages from 18-28, 29-38, etc. 
# 2. Then, i'm constructing 95% confidence intervals for the difference between the mean BMI for females and for males within each age band.

# In[59]:


np.max(da["RIDAGEYR"])


# In[60]:


da["agegrp_4"] = pd.cut(da.RIDAGEYR, [18, 28, 38, 48, 58, 68, 78, 88])
da_BMI = da.groupby(["agegrp_4", "RIAGENDRx"]).agg({"BMXBMI": [np.mean, np.std, np.size]}).unstack()


# In[64]:


da_BMI = da.groupby(["agegrp", "RIAGENDRx"]).agg({"BMXBMI": [np.mean, np.std, np.size]}).unstack()
print(da_BMI)

# Calculate the SEM for females and for males within each age band
da_BMI["BMXBMI", "sem", "Female"] = da_BMI["BMXBMI", "std", "Female"] / np.sqrt(da_BMI["BMXBMI", "size", "Female"]) 
da_BMI["BMXBMI", "sem", "Male"] = da_BMI["BMXBMI", "std", "Male"] / np.sqrt(da_BMI["BMXBMI", "size", "Male"])
print(da_BMI)

# Calculate the mean difference of BMI between females and males within each age band, also  calculate
# its SE and the lower and upper limits of its 95% CI.
da_BMI["BMXBMI", "mean_diff", ""] = da_BMI["BMXBMI", "mean", "Female"] - da_BMI["BMXBMI", "mean", "Male"]
da_BMI["BMXBMI", "sem_diff", ""] = np.sqrt(da_BMI["BMXBMI", "sem", "Female"]**2 + da_BMI["BMXBMI", "sem", "Male"]**2) 
da_BMI["BMXBMI", "lcb_diff", ""] = da_BMI["BMXBMI", "mean_diff", ""] - 1.96 * da_BMI["BMXBMI", "sem_diff", ""] 
da_BMI["BMXBMI", "ucb_diff", ""] = da_BMI["BMXBMI", "mean_diff", ""] + 1.96 * da_BMI["BMXBMI", "sem_diff", ""] 

# Plot the mean difference in black and the confidence limits in blue
x = np.arange(da_BMI.shape[0])
pp = sns.pointplot(x, da_BMI["BMXBMI", "mean_diff", ""], color='black')
sns.pointplot(x, da_BMI["BMXBMI", "lcb_diff", ""], color='blue')
sns.pointplot(x, da_BMI["BMXBMI", "ucb_diff", ""], color='blue')
pp.set_xticklabels(da_BMI.index)
pp.set_xlabel("Age group")
pp.set_ylabel("Female - male BMI difference")


# In[65]:


# Calculate the mean, SD, and sample size for BMI within age/gender groups
da_BMI = da.groupby(["agegrp_4", "RIAGENDRx"]).agg({"BMXBMI": [np.mean, np.std, np.size]}).unstack()
print(da_BMI)

# Calculate the SEM for females and for males within each age band
da_BMI["BMXBMI", "sem", "Female"] = da_BMI["BMXBMI", "std", "Female"] / np.sqrt(da_BMI["BMXBMI", "size", "Female"]) 
da_BMI["BMXBMI", "sem", "Male"] = da_BMI["BMXBMI", "std", "Male"] / np.sqrt(da_BMI["BMXBMI", "size", "Male"])
print(da_BMI)

# Calculate the mean difference of BMI between females and males within each age band, also  calculate
# its SE and the lower and upper limits of its 95% CI.
da_BMI["BMXBMI", "mean_diff", ""] = da_BMI["BMXBMI", "mean", "Female"] - da_BMI["BMXBMI", "mean", "Male"]
da_BMI["BMXBMI", "sem_diff", ""] = np.sqrt(da_BMI["BMXBMI", "sem", "Female"]**2 + da_BMI["BMXBMI", "sem", "Male"]**2) 
da_BMI["BMXBMI", "lcb_diff", ""] = da_BMI["BMXBMI", "mean_diff", ""] - 1.96 * da_BMI["BMXBMI", "sem_diff", ""] 
da_BMI["BMXBMI", "ucb_diff", ""] = da_BMI["BMXBMI", "mean_diff", ""] + 1.96 * da_BMI["BMXBMI", "sem_diff", ""] 

# Plot the mean difference in black and the confidence limits in blue
x = np.arange(da_BMI.shape[0])
pp = sns.pointplot(x, da_BMI["BMXBMI", "mean_diff", ""], color='black')
sns.pointplot(x, da_BMI["BMXBMI", "lcb_diff", ""], color='blue')
sns.pointplot(x, da_BMI["BMXBMI", "ucb_diff", ""], color='blue')
pp.set_xticklabels(da_BMI.index)
pp.set_xlabel("Age group")
pp.set_ylabel("Female - male BMI difference")


# Now I'm constructing a 95% confidence interval for the first and second systolic blood pressure measures, and for the difference between the first and second systolic blood pressure measurements within a subject.

# In[66]:


mean_1 = np.mean(da.BPXSY1)


# In[67]:


standev_1 = np.std(da.BPXSY1)


# In[68]:


total_1 = np.size(da.BPXSY1)


# In[69]:


stan_Error_1 = standev_1/np.sqrt(total_1)


# In[70]:


lcb_1 = mean_1 - 1.96 * stan_Error_1/ np.sqrt(total_1)
ucb_1 = mean_1 + 1.96 * stan_Error_1/ np.sqrt(total_1)
print(lcb_1, ucb_1)


# In[71]:


mean_2 = np.mean(da.BPXSY2)


# In[72]:


standev_2 = np.std(da.BPXSY2)


# In[73]:


total_2 = np.size(da.BPXSY2)


# In[74]:


stan_Error_2 = standev_2/np.sqrt(total_2)


# In[75]:


lcb_2 = mean_2 - 1.96 * stan_Error_2/ np.sqrt(total_2)
ucb_2 = mean_2 + 1.96 * stan_Error_2/ np.sqrt(total_2)
print(lcb_2, ucb_2)


# In[76]:


#difference in means 

se_dif = stan_Error_1 - stan_Error_2/np.sqrt(total_2)
d = mean_1 - mean_2
lcb_dif = d - 1.96*se_dif
ucb_dif = d + 1.96*se_dif
print(lcb_dif, ucb_dif)


# Finally, I'm constructing a 95% confidence interval for the mean difference between the average age of a smoker, and the average age of a non-smoker.

# In[77]:


da_smokers = da.groupby(["SMQ020x"]).agg({"RIDAGEYR": [np.mean, np.std, np.size]})
da_smokers


# In[78]:


da_smokers["RIDAGEYR", "sem"] = da_smokers["RIDAGEYR", "std"] / np.sqrt(da_smokers["RIDAGEYR", "size"]) 
da_smokers


# In[79]:


mean_Diff = da_smokers.loc["Yes", ("RIDAGEYR", "mean")] - da_smokers.loc["No", ("RIDAGEYR", "mean")]
sem_Diff = np.sqrt(da_smokers.loc["Yes", ("RIDAGEYR", "sem")]**2 + da_smokers.loc["No", ("RIDAGEYR", "sem")]**2) 
da_smokers["RIDAGEYR", "lcb"] = da_smokers.loc[:, ("RIDAGEYR", "mean")] - 1.96 * da_smokers.loc[:, ("RIDAGEYR", "sem")]
da_smokers["RIDAGEYR", "ucb"] = da_smokers.loc[:, ("RIDAGEYR", "mean")] + 1.96 * da_smokers.loc[:, ("RIDAGEYR", "sem")]


# In[80]:


lcb_Diff = mean_Diff - 1.96*sem_Diff
ucb_Diff = mean_Diff + 1.96*sem_Diff


# In[81]:


x = np.arange(da_smokers.shape[0])
pp = sns.pointplot(x, da_smokers["RIDAGEYR", "mean"], color='black')
sns.pointplot(x, da_smokers["RIDAGEYR", "lcb"], color='blue')
sns.pointplot(x, da_smokers["RIDAGEYR", "ucb"], color='blue')
pp.set_xticklabels(da_smokers.index)
pp.set_xlabel("smoker")
pp.set_ylabel("age")

