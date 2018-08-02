#
#
# Author: Dominique Cantave
# Project: TECBio 2018, University of Pittsburgh
# Mentors: Anne-Ruxandra Carvunis, David Koes
# fitness_graphs.py -> graphs for pGA performance testing
#
#

import glob
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_similarity_score
import matplotlib.pyplot as plt
import os
import random

""" DATA-RELATED PROCESSES """
# initializes the datasets
indexname = 'DJ'
year = 1925

stocks = pd.read_csv('stocklist.csv', sep=',', index_col=0)
stocks = stocks.iloc[:,0].values.tolist()
full_cols = ['in','age']
full_cols.extend(stocks)

# creates the comparison matrices as adjacency matrices
datafiles = glob.glob("Graphs/{}_{}/adjacency_matrix/range_*.csv".format(indexname,year))
correlations = []

for f in datafiles:
    a = int(f.split(os.path.sep)[-1].split('_')[1][0:-4])
    temp_df = pd.read_csv(f, sep=',', index_col=0)
    df = pd.DataFrame(index=stocks, columns=stocks)
    for i in temp_df.index.values:
        for j in temp_df.columns.values:
            df.loc[i][j] = int(temp_df.at[i,j])
    
    df.fillna(0,inplace=True)
    correlations.append(df)
    
# evolution time series [MIN, AVERAGE, MAX] fitnesses by generation
GA_init = pd.read_csv('GA_init_noP.csv', index_col=0, sep=',')
GA_noinit = pd.read_csv('GA_noinit.csv', index_col=0, sep=',')
GA2_init = pd.read_csv('GA2_init.csv', index_col=0, sep=',')
GA2_noinit = pd.read_csv('GA2_noinit.csv', index_col=0, sep=',')

# network with highest ending fitness
max_GA2_init = pd.read_csv('max_pop_GA2_init.csv', index_col=0, sep=',')
max_GA2_init = max_GA2_init.iloc[:,2:]
max_GA2_noinit = pd.read_csv('max_pop_GA2_noinit.csv', index_col=0,sep=',')
max_GA2_noinit = max_GA2_noinit.iloc[:,2:]
max_GA_init = pd.read_csv('max_pop_GA_noinit.csv', index_col=0,sep=',')
max_GA_init = max_GA_init.iloc[:,2:]
max_GA_noinit = pd.read_csv('max_pop_GA_noinit.csv', index_col=0,sep=',')
max_GA_noinit = max_GA_noinit.iloc[:,2:]

# network which has been in the population the longest
oldest_GA2_init = pd.read_csv('oldest_GA2_init.csv', index_col=0,sep=',')
oldest_GA2_init = oldest_GA2_init.iloc[:,2:]
oldest_GA2_noinit = pd.read_csv('oldest_GA2_noinit.csv', index_col=0,sep=',')
oldest_GA2_noinit = oldest_GA2_noinit.iloc[:,2:]
oldest_GA_noinit = pd.read_csv('oldest_GA_noinit.csv', index_col=0,sep=',')
oldest_GA_noinit = oldest_GA_noinit.iloc[:,2:]


def jaccard(df1, df2):
    
    """
    Calculates the element-wise Jaccard similarity of two DataFrames

    Parameters:
    :param df1, df2: DataFrames you want to compare

    :Returns: similarity of two DataFrame in the range [0,1]
    """
    
    dim1,dim2 = df1.shape
    tot=0
    k=0
    for i in range(dim1):
        for j in range(1,dim2):
            a = float(df1.iat[i,j])
            b = float(df2.iat[i,j])
            k += a*b
            tot += a+b>0
    
    return k/tot

# create the similarity sets
t = range(234)

# initial control
sims0 = []
for i in range(len(correlations)):
    sim = jaccard(correlations[i],correlations[0])
    sims0.append(sim)

# t-100 control
sims100 = []
for i in range(len(correlations)):
    sim = jaccard(correlations[i],correlations[100])
    sims100.append(sim)
    
# t-end control
sims233 = []
for i in range(len(correlations)):
    sim = jaccard(correlations[i],correlations[233])
    sims233.append(sim)
    
# max pGA without initialization
max2_no = []
for i in range(len(correlations)):
    sim = jaccard(correlations[i],max_GA2_noinit)
    max2_no.append(sim)

# max pGA with initialization 
max2_in = []
for i in range(len(correlations)):
    sim = jaccard(correlations[i],max_GA2_init)
    max2_in.append(sim)

# max GA with initialization
max1_in = []
for i in range(len(correlations)):
    sim = jaccard(correlations[i],max_GA_init)
    max1_in.append(sim)

# max GA without intialization
max1_no = []
for i in range(len(correlations)):
    sim = jaccard(correlations[i],max_GA_noinit)
    max1_no.append(sim)

# random control
simsrand = []
random.seed(10)
rand_df = pd.DataFrame(index=stocks,columns=stocks)
for i in stocks:
    for j in stocks:
        if i == j:
            rand_df.loc[i,j] = 0
        else:
            if random.random() < 0.2:
                rand_df.loc[i,j] = 1
                rand_df.loc[j,i] = 1
            else:
                rand_df.loc[i,j] = 0
            
for i in range(len(correlations)):
    sim = jaccard(correlations[i],rand_df)
    simsrand.append(sim)
    
# future control, similarity(n,n+1) timesteps
simsprev = []
for i in range(len(correlations)-1):
    sim = jaccard(correlations[i],correlations[i+1])
    simsprev.append(sim)

# oldest pGA with initialization
oldest2_in = []
for i in range(len(correlations)):
    sim = jaccard(correlations[i],oldest_GA2_init)
    oldest2_in.append(sim)
    
# oldest pGA without intialization
oldest2_no = []
for i in range(len(correlations)):
    sim = jaccard(correlations[i],oldest_GA2_noinit)
    oldest2_no.append(sim)

# oldest GA without intialization
oldest1_no = []
for i in range(len(correlations)):
    sim = jaccard(correlations[i],oldest_GA_noinit)
    oldest1_no.append(sim)
    
# oldest GA with initialization
oldest1_in = []
for i in range(len(correlations)):
    sim = jaccard(correlations[i],oldest_GA_init)
    oldest1_in.append(sim)


# plot against end time point
plt.plot(range(len(sims233)),sims233)
plt.title('relative to t=end')
plt.xlabel('timestep')
plt.ylabel('similarity')


# plot against start time point
plt.plot(range(len(sims0)),sims0)
plt.title('relative to t=start')
plt.xlabel('timestep')
plt.ylabel('similarity')


# plot against middle time point
plt.plot(range(len(sims100)),sims100)
plt.title('relative to t=middle')
plt.xlabel('timestep')
plt.ylabel('similarity')

# plot oldest pGA with init vs oldest GA with init vs initial control vs random control vs future control
plt.plot(t, oldest2_in, t,max1_in, t,sims0, t, simsrand, t[1:],simsprev)
plt.legend(['pGA with initial', 'GA with initial', 'initial control', 'random control', 'future control'])
plt.ylabel('similarity')
plt.xlabel('generations')
plt.title('initialized network similarities with control')

# plot oldest pGA without init vs oldest GA without init vs initial control vs random control vs future control
plt.plot(t, oldest2_no, t,oldest1_no, t, sims0, t, simsrand, t[1:],simsprev)
plt.legend(['pGA without initial', 'GA without initial','initial control', 'random control', 'future control'])
plt.ylabel('similarity')
plt.xlabel('generations')
plt.title('non-initialized network similarities with controls')
