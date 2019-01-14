# 
#
#
# Author: Dominique Cantave
# Project: TECBio 2018, University of Pittsburgh
# Mentors: Anne-Ruxandra Carvunis, David Koes
# pGA.py -> network growth simulation
# (with parallel computing the iterations can be split up)
#
#

import networkx as nx
from igraph import *
import numpy as np
from random import *
import matplotlib.pyplot as plt
import collections

# number of generations
GENS = 100

""" PARAMETERS TO SIMULATE """
# 1 = age-dependent, 0 = not age-dependent
MODE = [0,1]
# global rate of new nodes
BIRTHRATE = [0.2,0.6]
# rate relative to birthrate (x = x times lower)
DEATH = [5,10]
# rate of new edge gain
ER = [0.1,0.5,0.9]
# determines whether random edge loss is associated
EDGELOSS = [False,True]
# rate of new edges that are random (rather than preferential)
RANDRATE = [0.1,0.5,0.9]

# age dependency formula
def AGEDEP(age, baserate, mode):
    
    """
    Creates an optional weighting of a rate, here determined by age

    Parameters:
    :param age: age of the individual
    :param baserate: parameter to be weighted

    :Returns: weighted base parameter
    """
    
    ## NO DEPENDENCE ##
    if mode == 0:
        agepar = 0.5
        
    ## STRONG DEPENDENCE ##
    # sigmoid with age 10 arbitrarily chosen as new/old cutoff
    else:
        agepar = (-1*np.arctan(0.5*(age-10))/np.pi)+0.5

    # scaled base parameter
    return agepar*baserate


k = 30 # testing iterations
categories = ['# of edges', '# of nodes', '#components','average clustering'] # categories to be measured
cat_matrix = np.ndarray(shape=(len(categories),GENS,k))

"""
This script iterates through all combinations of parameters and measures number of nodes, edges, and
components of the graph, as well as connectivity and degree distribution. The goal is to determine 
what factors can change the make-up of the network.
"""
    
for mode in MODE:
    for br in BIRTHRATE:
        for death in DEATH:
            for er in ER:
                for el in EDGELOSS:
                    for rand in RANDRATE:
                        # for k iterations
                        for run in range(k):
                            """
                            Grows stochastic network

                            the pseudo-code goes as follows:
                                start with seed network (20 nodes with 50% random connectivity)
                                for GENS generations:
                                    add new node with one random intial edge
                                    increase age of all nodes
                                    remove zero-degree or killed nodes (death-rate relative to birth)
                                    add edges randomly or with preferential attachment
                                    randomly remove edge if edge loss is true
                                    
                                

                            :Returns: new individual network with age reset
                            """
                            
                            i=20
                            G = nx.binomial_graph(i, 0.5)
                            for node in G.nodes():
                                G.node[node]['age'] = 20

                            for j in range(GENS):
                                if random() < br:
                                    G.add_edge(i,choice(list(G.nodes())))
                                    G.node[i]['age'] = 0
                                    i+= 1

                                for node in list(G.nodes()):
                                    G.node[node]['age'] += 1
                                    age = G.node[node]['age']

                                    neighbors = list(G.neighbors(node))
                                    if not neighbors:
                                        G.remove_node(node)
                                        continue

                                    if random() < AGEDEP(age, br, mode)/(len(G.nodes())*death):
                                        G.remove_node(node)
                                        continue

                                    if random() < AGEDEP(age, er, mode):
                                        # attach edge to random other node
                                        if random() < rand:
                                            G.add_edge(node,choice(list(G.nodes())))
                                            
                                        # attach edge to a neighbor's neighbor
                                        else:
                                            # randomly picks one of the node's neighbors
                                            nghbr = choice(neighbors)
                                            nghbr2 = list(G.neighbors(nghbr))
                                            nghbr2.remove(node)
                                            if nghbr2: 
                                                G.add_edge(node, choice(nghbr2))
                                                
                                    if el and random() < AGEDEP(age, er, mode)/2:
                                        edgs = list(G.edges(node))
                                        G.remove_edge(*choice(edgs))

                                """
                                Calculates graph properties:
                                    (over time)
                                    number of edges
                                    number of nodes
                                    number of connected components
                                    average clustering coefficient
                                    (final)
                                    degree distribution
                                """
                                cat_matrix[0,j,run]=G.size()
                                cat_matrix[1,j,run]=len(G)
                                cat_matrix[2,j,run]=nx.number_connected_components(G)
                                cat_matrix[3,j,run]=nx.average_clustering(G)
                            
                            degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
                            if run == 0:    
                                degreeCount = collections.Counter(degree_sequence)
                            else:
                                degreeCount += collections.Counter(degree_sequence)
                            deg, cnt = zip(*degreeCount.items())
                            ct = tuple(c/k for c in cnt)

                        # mean and standard deviation for each timestep
                        cat_median = np.median(cat_matrix, axis=2)
                        cat_10p = np.percentile(cat_matrix,10,axis=2)
                        cat_90p = np.percentile(cat_matrix,90,axis=2)
                        
                        # creates the figure
                        fig = plt.figure(1,figsize=(10,10))
                        fig.clf()
                        
                        for i in range(len(categories)):
                            x = range(GENS)   # timesteps
                            y = cat_median[i,:]   # measurement mean
                            per10 = cat_10p[i,:]   # standard deviation
                            per90 = cat_90p[i,:]
                            
                            fig.add_subplot(2,1,i+1)
                            plt.plot(x, y, 'k', color='#2B2ACC')
                            plt.fill_between(x, y-error, y+error,
                                alpha=0.5, edgecolor='#4B2ACF', facecolor='#379FFF')
                            plt.title(categories[i])
                            plt.xlabel('generation')
                            plt.ylabel(categories[i])
                            plt.ylim(0.2,0.8)

                        fig.add_subplot(2,1,2)
                        plt.bar(deg, ct, width=0.85, color='b')
                        plt.title('final degree distribution')
                        plt.xlabel('degree')
                        plt.ylabel('nodes')
                        plt.ylim(0,10)
                        plt.xlim(0,50)
                        plt.tight_layout()

                        plt.savefig('Graphs/simulations/(B={},ER={},RR={},EL={},MODE={},D={}).pdf'.format(br,
                                 er, rand, el, mode, death), bbox_inches='tight')
                        plt.close(fig)

