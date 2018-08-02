
# coding: utf-8

# In[2]:


from igraph import *
import numpy as np
from random import *

GENS = 100
BIRTHRATE = 0.9
ER = 0.5
RANDRATE = 0.2

def AGEDEP(age, baserate):
    agepar = (-1*np.arctan(age-10)/np.pi)+0.5
    return agepar*baserate


# In[59]:


G = Graph.GRG(20, 0.5)
G.vs['age']=20
G.vs["label"] = g.vs["age"]

for j in range(1,GENS+1):
    
    # adds nodes according to birthrate   
    if random() < BIRTHRATE:
        #randomly attaches it to an existing node
        nodes = list(g.vs)
        G.add_edge(i,choice(list(g.vs)))
        G.node[i]['age'] = 0
        i+= 1


# In[60]:


layout = g.layout('kk')
plot(g, layout=layout)


# In[62]:


list(g.vs)

