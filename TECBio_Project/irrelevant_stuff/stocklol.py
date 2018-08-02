
# coding: utf-8

# In[1]:


from actual_work import *
import sys

stockset = sys.argv[1]
stockyear = stockset.split(os.path.sep)[-1].split('_')[2][0:4]
indexset = sys.argv[2]
indexname = indexset.split(os.path.sep)[-1].split('_')[0]

new_df, stkyr, idxname, dates = reform_data(stockset, indexset)

correlations = correlations(dates, new_df)

networks(correlations, indexname, stockyear, show_network=True)




# In[5]:


import itertools

before = [1, 8, 13]
after = [5, 11]
for i in itertools.izip_longest(before,after): print i


# In[8]:


import os
import pandas as pd
import numpy as np

stks = 'stocks_from_2018.csv'
idx = 'SP_constituents.csv'

stockyear = stks.split(os.path.sep)[-1].split('_')[2][0:4]
indexname = idx.split(os.path.sep)[-1].split('_')[0]

# reading / organizing the datasets
stockslist = pd.read_csv(stks)
dates = stockslist['date'].unique()

index = pd.read_csv(idx)
index = index.set_index(['co_tic'])
index.sort_index(level='from',inplace=True)

indexlist = stockslist[stockslist['TICKER'].isin(index.index.values)]
indexlist = indexlist.set_index(['TICKER', 'date'])
indexlist = indexlist[~indexlist.index.duplicated(keep='last')]
stocks = np.unique(indexlist.index.get_level_values(0))
stocks = stocks[~pd.isnull(stocks)]

stocks


# In[16]:


from itertools import *

new_df = pd.DataFrame(index=stocks) # new df for storing price information
for stock in stocks:

    # creates list of ranges corresponding to dates that the stock was in the index
    # if stock was removed and added again, ranges has more than 1 element
    fromdate = as_list(index.at[stock,'from'])
    thrudate = as_list(index.at[stock,'thru'])
    ranges = list(izip_longest(fromdate,thrudate))

    # considers only the dates within these ranges
    for ran in ranges:
        for date in indexlist.loc[stock].index.values:
            # makes sure to only account for dates since it was added
            if date >= ran[0]:
                # and only go until the date removed if applicable
                if np.isnan(ran[-1]) or date <= ran[-1]:
                    new_df.at[stock, date] = indexlist.at[(stock,date),'PRC']

new_df = new_df.dropna(how='all') 
new_df = new_df.T
new_df.loc[20180105, 'BA'] = None
new_df = new_df.pct_change() 
dates = np.sort(new_df.index.unique())


# In[49]:





# In[10]:


def as_list(x):
    if type(x) is np.ndarray:
        return x.tolist()
    else:
        return [x]


# In[10]:


value = False
value = not value
value


# In[17]:


0 == None

