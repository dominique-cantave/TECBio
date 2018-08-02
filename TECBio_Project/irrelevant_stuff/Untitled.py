
# coding: utf-8

# In[2]:


import networkx as nx
from igraph import *
import numpy as np
from random import *
import matplotlib.pyplot as plt
import collections


# In[8]:


def AGEDEP(age, baserate, mode):
    if mode == 0:
        agepar = 0.5 # not dependent
        
    # sigmoid with age 10 arbitrarily chosen as new/old cutoff
    else:
        agepar = (-1*np.arctan(0.5*(age-10))/np.pi)+0.5  # slight dependence  

    # rate relative to base parameter
    return agepar*baserate

GENS = 100
k = 30
cat_matrix = np.ndarray(shape=(GENS,k))
cat_matrix2 = np.ndarray(shape=(GENS,k))


# In[40]:


for run in range(k):
    # create initial population of 20 genes with random edges
    i=20
    G = nx.binomial_graph(i, 0.5)
    for node in G.nodes():
        G.node[node]['age'] = 20

    for j in range(GENS):

        # adds nodes according to birthrate   
        if random() < 0.6:
            #randomly attaches it to an existing node
            G.add_edge(i,choice(list(G.nodes())))
            G.node[i]['age'] = 0
            i+= 1

        for node in list(G.nodes()):
            # increases the age of each node
            G.node[node]['age'] += 1
            age = G.node[node]['age']

            # removes unattached nodes
            neighbors = list(G.neighbors(node))
            if not neighbors:
                G.remove_node(node)
                continue

            # decides if it dies with total deathrate related to birthrate
            if random() < AGEDEP(age, 0.6, 0)/(len(G.nodes())*10):
                G.remove_node(node)
                continue

            # adds edges according to some scheme
            if random() < AGEDEP(age, 0.5, 0):

                # random attachment
                if random() < 0.5:
                    G.add_edge(node,choice(list(G.nodes())))

                # preferential attachment
                else:
                    # randomly picks one of the node's neighbors
                    nghbr = choice(neighbors)
                    # goes through nghbr's own neighbors (excluding the node itself)
                    nghbr2 = list(G.neighbors(nghbr))
                    nghbr2.remove(node)
                    # if the nghbr has other neighbors, randomly assoicate with one
                    if nghbr2: 
                        G.add_edge(node, choice(nghbr2))

            # randomly removes an edge from the node
#             if el and random() < AGEDEP(age, er, mode)/2:
#                 edgs = list(G.edges(node))
#                 G.remove_edge(*choice(edgs))

        #### adjusts the category measurement matrix with each
        # number of connected subgraphs
        cat_matrix[j,run]=nx.average_clustering(G)   # averages the local clustering for each node
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    if run == 0:    
        degreeCount = collections.Counter(degree_sequence)
    else:
        degreeCount += collections.Counter(degree_sequence)
    deg2, cnt2 = zip(*degreeCount.items())
    ct2 = tuple(c/k for c in cnt2)


# In[16]:


cat_median1 = np.median(cat_matrix, axis=1)
cat_10p1 = np.percentile(cat_matrix,10,axis=1)
cat_90p1 = np.percentile(cat_matrix,90,axis=1)


# In[14]:


cat_median2 = np.median(cat_matrix2, axis=1)
cat_10p2 = np.percentile(cat_matrix2,10,axis=1)
cat_90p2 = np.percentile(cat_matrix2,90,axis=1)


# In[17]:


get_ipython().magic(u'matplotlib qt')
x = range(GENS)   # timesteps
y = cat_median1   # measurement mean
z = cat_median2
errorup = cat_90p1
errorup2 = cat_90p2# standard deviation
errordown = cat_10p1
errordown2 = cat_10p2

plt.plot(x, y, x,z)
plt.fill_between(x, errordown, errorup, alpha=0.5)
plt.fill_between(x,errordown2, errorup2, alpha=0.5)
plt.title('average clustering coefficient')
plt.xlabel('generation')
plt.ylabel('average clustering')
plt.ylim(0.2,0.8)
plt.legend(['with age dependence', 'without age dependence'])


# In[59]:


get_ipython().magic(u'matplotlib qt')
plt.style.use('seaborn-deep')

plt.bar(np.array(deg)-0.2, ct, width=0.4, label='with age dependence')
plt.bar(np.array(deg2)+0.2, ct2, width=0.4, label='without age dependence')
plt.title('final degree distribution')
plt.xlabel('degree')
plt.ylabel('nodes')
plt.ylim(0,9)
plt.xlim(0,45)
plt.legend(['with age dependence','without age dependence'])


# In[41]:


#plt.bar(deg, ct, label='with age dependence')
plt.bar(deg2, ct2, label='without age dependence')
plt.title('final degree distribution')
plt.xlabel('degree')
plt.ylabel('nodes')
plt.ylim(0,9)
plt.xlim(0,50)


# In[49]:


type(deg2)

