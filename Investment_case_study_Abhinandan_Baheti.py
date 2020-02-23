#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


# Checkpoint 1 : Understand the Data Set  : Table-1.1
# Loading csv files
companies = pd.read_csv('companies.csv',delimiter="\t", encoding = 'palmos') 
# Checkpoint 1 : Understand the Data Set  : Table-1.1 (question 2)
# Unique companies present in companies file
companies['permalink'].str.upper().nunique()


# In[3]:


# Checkpoint 1 : Understand the Data Set  : Table-1.1
# Loading csv files
rounds2 = pd.read_csv("rounds2.csv", encoding='ISO-8859-1')
# Checkpoint 1 : Understand the Data Set  : Table-1.1 (question 1) 
# Unique companies present in rounds2 file
rounds2['company_permalink'].str.upper().nunique()


# In[4]:


# Checkpoint 1 : Understand the Data Set  : Table-1.1 (question 3)
# To capture unique column
companies.nunique()


# In[5]:


# Checkpoint 1 : Understand the Data Set  : Table-1.1 (question 3)
# To capture unique column
companies.count()


# In[6]:


rounds2.nunique()


# In[7]:


# Checkpoint 1 : Understand the Data Set  : Table-1.1 (question 4)
# rounds2 companies which are not present in Companies
rounds2[~rounds2['company_permalink'].str.upper().isin(companies['permalink'].str.upper())]


# In[8]:


# Checkpoint 1 : Understand the Data Set  : Table-1.1
companies['permalink'] = companies['permalink'].str.lower()
rounds2['company_permalink'] = rounds2['company_permalink'].str.lower()


# In[9]:


# Checkpoint 1 : Understand the Data Set  : Table-1.1 (question 5)
# Merging Companies and rounds2 dataframe with outer join
master_frame = pd.merge(companies, rounds2, left_on='permalink', right_on='company_permalink', how='inner') 
# Checkpoint 1 : Understand the Data Set  : Table-1.1 (question 5)
# Total rows/cols after merge with outer join and igonring cases of permalink and company_permalink
master_frame.shape


# In[10]:


# Checkpoint 1 : Understand the Data Set  : Table-1.1 
companies.shape


# In[11]:


# Checkpoint 1 : Understand the Data Set  : Table-1.1
rounds2.shape


# In[12]:


master_frame.head()


# In[13]:


# capturing null values sum across all columns
round(100*(master_frame.isnull().sum()/len(master_frame.index)), 2)


# In[14]:


# Removing rows with raised amount usd value as null
master_frame = master_frame[~np.isnan(master_frame['raised_amount_usd'])]
# Removing non usable rows and duplicate column - company permalink
master_frame = master_frame.drop('funding_round_code', axis=1)
master_frame = master_frame.drop('company_permalink', axis=1)
master_frame = master_frame.drop('state_code', axis=1)
master_frame = master_frame.drop('region', axis=1)
master_frame = master_frame.drop('founded_at', axis=1)
master_frame = master_frame.drop('homepage_url', axis=1)
master_frame = master_frame.drop('city', axis=1)
round(100*(master_frame.isnull().sum()/len(master_frame.index)), 2)


# In[15]:


# Imputing NA for country_code and category_list
master_frame.loc[pd.isnull(master_frame['category_list']), ['category_list']] = 'NA'
master_frame.loc[pd.isnull(master_frame['country_code']), ['country_code']] = 'NA'
round(100*(master_frame.isnull().sum()/len(master_frame.index)), 2)


# In[16]:


# Almost 18% rows is cleaned up
len(master_frame.index)/114919


# In[17]:


# Checkpoint 2 :  Average Values of Investments for Each of these Funding Types  : Table-2.1 
#Table 2.1: Grouping by funding round type
seg = master_frame.groupby('funding_round_type')
#Table 2.1: Setting option to display funding amount in non scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#Table 2.1: Capturing Average Values of Investments for Each of these Funding Types (Question 1,2,3,4,5)
seg['raised_amount_usd'].mean()


# In[18]:


# Checkpoint 3 :  Analysing the Top 3 English-Speaking Countries  : Table-3.1 
top9 = master_frame[master_frame['funding_round_type'] == 'venture']
seg_country_code = top9.groupby('country_code')
seg_country_code.raised_amount_usd.sum().sort_values( ascending = False)


# In[19]:


#Checkpoint 4: Sector Analysis 1
# Extracting first param in category list and placing in primary_cector
master_frame['primary_sector'] = master_frame['category_list'].str.split("|", n = 1, expand = True)[0]
master_frame.head()


# In[20]:


#Checkpoint 4: Sector Analysis 1 - Mapping primary_sector to main_sector from mapping.csv file
data = pd.read_csv("mapping.csv", encoding='ISO-8859-1')  
dict_sector_mapping = {}
for key, value in data.iterrows():
    val = ''
    if key == 0:
        continue
    for x,y in zip(value.index, value._values):
        if x == 'category_list':
            val = y
        if y == 1:
            dict_sector_mapping[val] = x

sector_df = pd.DataFrame(list(dict_sector_mapping.items()), columns=['primary_sector', 'main_sector'])
sector_df.head()


# In[21]:


#Checkpoint 4: Sector Analysis 1 
# Joining sector and master frame to get main_Sector for each primary_sector
master_frame_new = pd.merge(master_frame, sector_df, on='primary_sector', how='inner') 
master_frame_new.head()


# In[22]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Capturing raised amount usd in millions
master_frame_new['raised_amount_usd_in_millions'] = (master_frame_new['raised_amount_usd'].astype(float)/1000000).astype(float)
master_frame_new.head()


# In[23]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Finding out venture with Country code USA and funding between 5-15m
master_new = master_frame_new[(master_frame_new['funding_round_type']=='venture') & (master_frame_new['country_code'] == 'USA') 
                      &( master_frame_new['raised_amount_usd_in_millions'] >= 5) & ( master_frame_new['raised_amount_usd_in_millions'] <= 15)]
master_new.head()


# In[24]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Capturing total funding amount for each main_sector
D1_TEMP = master_new.groupby('main_sector').raised_amount_usd.sum()
D1_TEMP


# In[25]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Capturing total investment for all main sectors for USA
D1_TEMP1 = D1_TEMP.reset_index()
D1_TEMP1.rename(columns = {"raised_amount_usd": "Total Investments"}, inplace = True)
D1_TEMP1


# In[26]:


# Sum of total investment for all main sectors for USA
D1_TEMP1['Total Investments'].sum()


# In[27]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Merging with main dataframe
D1 = pd.merge(master_new, D1_TEMP1, on='main_sector', how='outer')
D1.head()


# In[28]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Grouping by main sector and capturig total count of investments
D1_TEMP2 = D1.groupby('main_sector').raised_amount_usd.count()
D1_TEMP3 = D1_TEMP2.reset_index()
D1_TEMP3


# In[29]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Sum of number of investments
D1_TEMP3.rename(columns = {"raised_amount_usd": "Count (Investments)"}, inplace = True)
D1_TEMP3['Count (Investments)'].sum()


# In[30]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Merging with main dataframe having both columns count(Investments) + Total Investments for each main sector for Country 1 - USA
D1 = pd.merge(D1, D1_TEMP3, on='main_sector', how='outer') 
# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 ** D1 Dataframe ** 
D1.head()


# In[31]:


# Sorting for USA - On basis of number of investments
D1.sort_values(by='Count (Investments)', ascending=False)


# In[32]:


# Sorting for USA - On basis of number of investments for second highest sector values as 2297
D1_TEMP4 = D1.sort_values(by='Count (Investments)', ascending=False)
D1_TEMP4.loc[D1_TEMP4['Count (Investments)'] == 2297]


# In[33]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Finding out venture with Country code GBR and funding between 5-15m
master_new_d2 = master_frame_new[(master_frame_new['funding_round_type']=='venture') & (master_frame_new['country_code'] == 'GBR') 
                      &( master_frame_new['raised_amount_usd_in_millions'] >= 5) & ( master_frame_new['raised_amount_usd_in_millions'] <= 15)]


# In[34]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Capturing total funding amount for each main_sector
D2_TEMP = master_new_d2.groupby('main_sector').raised_amount_usd.sum()
D2_TEMP1 = D2_TEMP.reset_index()
D2_TEMP1.rename(columns = {"raised_amount_usd": "Total Investments"}, inplace = True)
D2_TEMP1.head()


# In[35]:


# Total sum of investments
D2_TEMP1['Total Investments'].sum()


# In[36]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Merging with main dataframe
D2 = pd.merge(master_new_d2, D2_TEMP1, on='main_sector', how='outer')
D2.head()


# In[37]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Grouping by main sector and capturig total count of investments
D2_TEMP2 = D2.groupby('main_sector').raised_amount_usd.count()
D2_TEMP3 = D2_TEMP2.reset_index()
D2_TEMP3


# In[38]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# sum of total number of investments
D2_TEMP3.rename(columns = {"raised_amount_usd": "Count (Investments)"}, inplace = True)
D2_TEMP3['Count (Investments)'].sum()


# In[39]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Merging with main dataframe having both columns count(Investments) + Total Investments for each main sector for Country 1 - USA
D2 = pd.merge(D2, D2_TEMP3, on='main_sector', how='outer') 
D2.head()


# In[40]:


# Sorting for GBR - on basis on number of investments
D2.sort_values(by='Count (Investments)', ascending=False)


# In[41]:


# # Sorting for GBR - On basis of number of investments for second highest sector values as 127
D2_TEMP4 = D2.sort_values(by='Count (Investments)', ascending=False)
D2_TEMP4.loc[D2_TEMP4['Count (Investments)'] == 127]


# In[42]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Finding out venture with Country code IND and funding between 5-15m
master_new_d3 = master_frame_new[(master_frame_new['funding_round_type']=='venture') & (master_frame_new['country_code'] == 'IND') 
                      &( master_frame_new['raised_amount_usd_in_millions'] >= 5) & ( master_frame_new['raised_amount_usd_in_millions'] <= 15)]


# In[43]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Capturing total funding amount for each main_sector
D3_TEMP = master_new_d3.groupby('main_sector').raised_amount_usd.sum()
D3_TEMP1 = D3_TEMP.reset_index()
D3_TEMP1.rename(columns = {"raised_amount_usd": "Total Investments"}, inplace = True)
D3_TEMP1.head()


# In[44]:


# total sum of investments for IND
D3_TEMP1['Total Investments'].sum()


# In[45]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Merging with main dataframe
D3 = pd.merge(master_new_d3, D3_TEMP1, on='main_sector', how='outer')
D3.head()


# In[46]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Grouping by main sector and capturig total count of investments
D3_TEMP2 = D3.groupby('main_sector').raised_amount_usd.count()
D3_TEMP3 = D3_TEMP2.reset_index()
D3_TEMP3


# In[47]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# TOtal sum of number of investments for IND
D3_TEMP3.rename(columns = {"raised_amount_usd": "Count (Investments)"}, inplace = True)
D3_TEMP3['Count (Investments)'].sum()


# In[48]:


# Checkpoint 5 :  Sector-wise Investment Analysis  : Table-5.1 
# Merging with main dataframe having both columns count(Investments) + Total Investments for each main sector for Country 1 - USA
D3 = pd.merge(D3, D3_TEMP3, on='main_sector', how='outer') 
D3.head()


# In[49]:


# Sorting for IND - on basis on number of investments
D3.sort_values(by='Count (Investments)', ascending=False)


# In[50]:


# Sorting for IND - On basis of number of investments for second highest sector values as 52
D3_TEMP4 = D3.sort_values(by='Count (Investments)', ascending=False)
D3_TEMP4.loc[D3_TEMP4['Count (Investments)'] == 52]


# In[51]:


# Checkpoint 6 :  Plots
# Extracting 'venture', 'angel', 'seed', 'private_equity'
master_frame_new1 = master_frame_new[master_frame_new['funding_round_type'].isin(['venture', 'angel', 'seed', 'private_equity'])]


# In[52]:


# Capturing fractions of investments for funding types - 'venture', 'angel', 'seed', 'private_equity'
fractions_frame = master_frame_new1.groupby('funding_round_type').agg({'raised_amount_usd': 'sum'})
fractions_frame = fractions_frame.apply(lambda x:
                                                 100 * x / float(master_frame_new1['raised_amount_usd'].sum()))
fractions_frame.reset_index(inplace = True) 
fractions_frame.rename(columns = {"raised_amount_usd": "percentage_investments"}, inplace = True)
fractions_frame.head()


# In[53]:


# Checkpoint 6  Plot 1 - A plot showing the fraction of total investments (globally) in venture, seed, and private equity, 
#and the average amount of investment in each funding type
#plt.subplots(1, 3)
plt.figure(figsize=(20, 6))
plt.subplot(1, 3, 1)
ax = sns.barplot(x="funding_round_type", y='raised_amount_usd', data=master_frame_new1, estimator=np.sum)
for p in ax.patches:
         ax.annotate('$'+format(p.get_height()/1000000000, '.2f')+'B', (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 13), textcoords = 'offset points')
            
plt.yscale('log')
plt.title("Total Investments (In Billions)")

plt.subplot(1, 3, 2)
ax = sns.barplot(x="funding_round_type", y='raised_amount_usd', data=master_frame_new1, estimator=np.mean)
#total = len(master_frame_new1['raised_amount_usd'])
plt.title("Average Investment (In Millions)")
plt.yscale('log')
for p in ax.patches:
         ax.annotate('$'+format(p.get_height()/1000000, '.2f') + 'M', (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 15), textcoords = 'offset points')

plt.subplot(1, 3, 3)
ax = sns.barplot(x="funding_round_type", y='percentage_investments', data=fractions_frame, estimator=np.mean)
for p in ax.patches:
         ax.annotate(format(p.get_height(), '.2f') + '%', (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')
plt.title("Fraction of Investments")
plt.show()


# In[54]:


# Extracting 'venture' type only
master_frame_new2 = master_frame_new1[master_frame_new1['funding_round_type'] == 'venture']


# In[55]:


master_frame_new2


# In[56]:


# Extracting Top 9 Countries
top9_countries = master_frame[master_frame['country_code'].isin(['USA', 'CHN', 'GBR', 'IND', 'CAN', 'FRA', 'ISR', 'DEU', 'JPN' ])]


# In[57]:


top9_countries


# In[58]:


# Checkpoint 6 Plot 2 - A plot showing the top 9 countries against the total amount of investments of funding type FT - Venture
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
ax = sns.countplot(x="country_code", data=top9_countries)
for p in ax.patches:
         ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title("Count of Investments")
plt.subplot(1, 2, 2)
ax = sns.barplot(x="country_code", y="raised_amount_usd", data=top9_countries, estimator=np.sum)
for p in ax.patches:
         ax.annotate('$'+format(p.get_height()/1000000000, '.2f') + 'B', (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 25), textcoords = 'offset points')

plt.title("Total Investments (In Billion dollars)")
plt.yscale('log')
plt.show()


# In[59]:


# Checkpoint 6 : A plot showing the number of investments in the all sectors of the top 3 countries on one chart
frames = [D1, D2, D3]
final_frame = pd.concat(frames)
plt.figure(figsize=(20, 10))
ax = sns.barplot(x='country_code', y='Count (Investments)', hue="main_sector", data=final_frame, estimator=np.mean)
for p in ax.patches:
         ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 15), textcoords = 'offset points')
        
plt.title('Total Investments across all Sector')
plt.show()


# In[60]:


D1_T = D1[D1['main_sector'].isin(['Others', 'Cleantech / Semiconductors', 'Social, Finance, Analytics, Advertising'])]
D2_T = D2[D2['main_sector'].isin(['Others', 'Cleantech / Semiconductors', 'Social, Finance, Analytics, Advertising'])]
D3_T = D3[D3['main_sector'].isin(['Others', 'News, Search and Messaging', 'Entertainment'])]


# In[61]:


D1_T


# In[62]:


# Checkpoint 6 : Plot 3 -  A plot showing the number of investments in the top 3 sectors of the top 3 countries on one chart
frames = [D1_T, D2_T, D3_T]
final_frame = pd.concat(frames)
plt.figure(figsize=(20, 10))
ax = sns.barplot(x='country_code', y='Count (Investments)', hue="main_sector", data=final_frame, estimator=np.mean)
for p in ax.patches:
         ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 15), textcoords = 'offset points')

plt.title('No of Investments in Top 3 Sector')
plt.show()


# In[ ]:




