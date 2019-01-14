# 
#
#
# Author: Dominique Cantave
# Project: TECBio 2018, University of Pittsburgh
# Mentors: Anne-Ruxandra Carvunis, David Koes
# pGA.py -> run proto-genetic algorithm with changing fitness function
#
#
#

import glob
import os
import pandas as pd
import numpy as np
import sys
import random
from sklearn.metrics import accuracy_score
from itertools import repeat
from operator import itemgetter

from deap import base, creator, tools, algorithms

""" MISCELLANEOUS FUNCTIONS: """
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

def AGEDEP(age, baserate):
        
    """
    Creates an optional weighting of a rate, here determined by age

    Parameters:
    :param age: age of the individual
    :param baserate: parameter to be weighted

    :Returns: weighted base parameter
    """
    
    ## STRONG DEPENDENCE ##
    # sigmoid with age 50 arbitrarily chosen as new/old cutoff
    agepar = (-1*np.arctan(0.5*(age-100))/np.pi)+0.5
    ## NO DEPENDENCE ##
    # agepar = 1
    
    return agepar*baserate


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
comparisons = []

for f in datafiles:
    temp_df = pd.read_csv(f, sep=',', index_col=0)
    df = pd.DataFrame(index=stocks, columns=['in']+stocks)
    for i in temp_df.index.values:
        for j in temp_df.columns.values[1:]:
            df.loc[i, j] = int(temp_df.at[i,j])
        
        if df.loc[i,stocks].sum() != 0:
            df.loc[i,'in'] = 1
        else:
            df.loc[i,'in'] = 0
            
    df.fillna(0,inplace=True)
    
    # allows three generations for the algorithm to train to each dataset
    for _ in range(3):
        comparisons.append(df)


def init_individual(index, columns, initializer=None):
    
    """
    Initializes the networks belonging to each individual as a pandas DataFrame. This is what 
    the similarities are calculated against.

    Parameters:
    :param index: the index names that you want
    :param columns: the column names you want
    :param initializer: determines if individuals are random or 
    initialized; if True, initialized from the initializer input, 
    else random

    :Returns: pandas DataFrame
    """
    
    ind = pd.DataFrame(0,index=index, columns=columns)
    
    if initializer is not None:
        
        # sets up the DataFrame with the initializer data
        ind.loc[:, 2:] = initializer.loc[:,1:]
        ind.loc[:, 'in'] = initializer.loc[:, 'in']
        
        # sets the age
        for i in index:
            if ind.loc[i,'in'] != 0:
                ind.loc[i,'age'] = 1
            else:
                ind.loc[i,'age'] = 0
                
            # randomly flips a company in or out of the system
            if random.random() < 0.05:
                if ind.loc[i,'in'] == 0:
                    ind.loc[i,'in'] = 1
                    ind.loc[i,'age'] = 1
                    for j in index:
                        if i == j:
                            ind.loc[i,j] = 0
                        else:
                            if random.random() < 0.2:
                                ind.loc[i,j] = 1
                                ind.loc[j,i] = 1
                else:
                    ind.loc[i,:] = 0
                    ind.loc[:,i] = 0
                    
            # randomly flips correlations
            if ind.loc[i,'in'] == 1:        
                for j in index:
                    if random.random() < 0.05 and i != j:
                        ind.loc[i,j] = abs(ind.loc[i,j] - 1)
                        ind.loc[j,i] = ind.at[i,j]

    else:
        for i in index:
            # randomly places companies in or out of the network
            if random.random() < 0.2:
                ind.loc[i,'in'] = 1
                ind.loc[i,'age'] = 1
                
            # randomly assigns correlations for companies in the network
            if ind.loc[i,'in'] == 1:
                for j in index:
                    if i == j:
                        ind.loc[i,j] = 0
                    else:
                        if random.random() < 0.2
                        ind.loc[i,j] = 1
                        ind.loc[j,i] = ind.at[i,j]
                
        ind.fillna(0)

    return ind

class Individual(object):
    
    """
    Creates the individual class.

    init parameters:
    :param index: the index names that you want
    :param columns: the column names you want
    :param initializer: determines if individuals are random or 
    initialized; if True, initialized from the initializer input, 
    else random 
                      
    Attributes:
    :attr network: adjacency network
    :attr age: age of each individual

    :Returns: list of n-sized chunks
    """
    
    def __init__(self, index, columns, initializer=None):
        self.network = init_individual(index,columns,initializer)
        self.age = 1      
        
# structuring initializers
# FitnessMax => weights fitness by whoever is highest
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

## WITHOUT INITIALIZATION ##
creator.create("Individual", Individual, fitness=creator.FitnessMax)
## WITH INITIALIZATION ##
# creator.create("Individual", Individual, fitness=creator.FitnessMax, comparisons[0])


toolbox = base.Toolbox()2
        
# creates the population
toolbox.register("individual", creator.Individual, stocks, full_cols, comparisons[0])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def eval_fitness(individual, comparison):
    
    """
    Evaluates the fitness of each individual, in terms of similarity.

    Parameters:
    :param individual: the individual to be evaluated
    :param comparison: the reference that similarity will be 
    calculated against
    
    :Returns: float with similarity measure in range [0,1]
    """
    
    adjacency = individual.network.loc[:,['in']+stocks]
    for i in individual.network.index:
        individual.network.loc[i,'age'] += 1
    return jaccard(adjacency,comparison),

def mut_individual(individual, pexist):
    
    """
    Mutates the individual network

    init parameters:
    :param individual: individual to be mutated
    :param pexist: the probability of adding/removing a company from 
    the network
    
    the pseudo-code goes as follows:
        for each stock, with probability pexist switch in/out
            if now out, reset every thing to 0
            if now in, randomly initialize adjacencies and set age=1
            
        choose a 10 adjacency terms from stocks that are in, and 
        changing them
        reset the ages

    :Returns: new individual network with age reset
    """
    
    network = individual.network
    for i in network.index.values:
        age = network.loc[i,'age']
        if random.random() < AGEDEP(age, pexist):
            if network.loc[i,'in'] == 1:
                network.loc[i, :] = 0
                network.loc[:, i] = 0
                
            if network.loc[i,'in'] == 0:
                network.loc[i,'in'] = 1
                network.loc[i,'age'] = 1
                for j in network.columns.values[2:]:
                    if random.random() < 0.1 and i != j:
                        network.loc[i,j] = 1
                        network.loc[j,i] = network.at[i,j]
                    
    relevant = network.loc[network['in']==1]
    for _ in range(10):
        i = random.choice(relevant.index.values)
        j = random.choice(relevant.columns.values[2:])
        network.loc[i,j] = abs(network.at[i,j]-1)
        network.loc[j,i] = network.at[i,j]
                
    if network.loc[i][1:].sum() == 0:
        network.loc[i,'in'] = 0    
        network.loc[i,'age'] = 0
        
    individual.network = network
    individual.age = 1
    return individual,

def cross_over(ind1, ind2):
    
    """
    Crossing-over with 2 individuals

    init parameters:
    :param ind1: individual 1
    :param ind2: individual 2
    
    the pseudo-code goes as follows:
        randomly choose a crossing-over point
        swap elements [0:cx,0:cx] from the 2 individuals
        reset the ages

    :Returns: new individual network with age reset
    """
    
    network1 = ind1.network
    network2 = ind2.network
    
    size = min(len(network1.index), len(network2.index))
    cx = random.randint(1, size - 1)
    
    temp = network1.copy()
    temp.iloc[:cx,:cx] = network2.iloc[:cx,:cx]
    network2.iloc[:cx,:cx] = network1.iloc[:cx,:cx]
    network1 = temp 
    
    ind1.network = network1
    ind2.network = network2
    ind1.age = 1
    ind2.age = 1
    
    return ind1, ind2
            
# set the DEAP equations
toolbox.register("evaluate", eval_fitness)
toolbox.register("mutate", mut_individual, pexist=0.3)
toolbox.register("select", tools.selBest)
toolbox.register("select2", tools.selTournament, tournsize=4)
toolbox.register("mate", cross_over)


def main():
    
    """
    Genetic algorithm

    the pseudo-code goes as follows:
        initialize the population and parameters
        initialize the statistics desired
        GA_algorithm:
            calculate fitnesses and print statistics
            for NGEN generations:
                create the offspring
                re-evaluate fitnesses and select population
                update ages of inidividuals
                print statistics
                
            return to max-fitnesses and oldest individuals at the end

    :Returns: new individual network with age reset
    """
    
    NGEN = len(comparisons)
    MU = 50
    CXPB = 0.5
    MUTPB = 0.5

    pop = toolbox.population(n=MU)
#     hof = tools.HallOfFame(15)
    hof=None
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
#     stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    def GA_algorithm(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
        
        # original algorithm
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        fitnesses = toolbox.map(toolbox.evaluate, population, [comparisons[0]]*len(population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(population), **record)
        
        if verbose:
	    print(logbook.stream)

        for g in range(1,NGEN):
            # Vary the pool of individuals
            offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness        
            fitnesses = toolbox.map(toolbox.evaluate, population+offspring, [comparisons[g]]*len(population+offspring))
            for ind, fit in zip(population+offspring, fitnesses):
                ind.fitness.values = fit
            if halloffame is not None:
                halloffame.update(offspring)

            population[:10] = toolbox.select(population+offspring, 10)
            population[10:] = toolbox.select2(population+offspring, 40)
		
            for ind in population:
                ind.age += 1

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=g, nevals=len(population+offspring), **record)
            if verbose:
                print(logbook.stream)
 
        ages = [ind.age for ind in population]
        max_age = max(ages)
        oldest_pop = [i.network for i in population if i.age == max_age]
        oldest_pop[0].to_csv('oldest_GA_init.csv')

        max_fit = max(fitnesses)
        max_pop = [i.network for i in population if i.fitness.values == max_fit]
        max_pop[0].to_csv('max_pop_GA_init.csv')
    
        return population, logbook
    
    GA_algorithm(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats,
                        halloffame=hof)
    
    return pop, stats, hof

if __name__ == "__main__":
    main()

