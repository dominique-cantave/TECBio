# 
#
#
# Author: Dominique Cantave
# Project: TECBio 2018, University of Pittsburgh
# Mentors: Anne-Ruxandra Carvunis, David Koes
# processing.py -> create correlation network from input data
#
#
#


import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os
import glob
import seaborn
from operator import itemgetter
from itertools import groupby, izip_longest

""" MISCELLANEOUS FUNCTIONS: """


def grouper(iterable, n, fillvalue=None):
    """
    Increments the data into n-sized chunks

    Parameters:
    :param iterable: The data that you want to iterate through; must be iter() compatible
    :param n: number of values you want in each chunk; must be an integer
    :param fillvalue: if length of iterable is not divisible by n, the last chunk will 
                      be filled with the fillvalue to maintain size

    :Returns: list of n-sized chunks
    """
    args = [iter(iterable)] * n
    group = list(izip_longest(*args, fillvalue=fillvalue))
    ranges = []
    for i in range(len(group)):
        ranges.append([group[i][0],group[i][-1]])
    return ranges

def as_list(x):
    """
    Casts array or integer to list

    Parameters:
    :param x: The object that you want to turn into a list; must be ndarray or int

    :Returns: list
    """
    if type(x) is np.ndarray:
        return x.tolist()
    else:
        return [x]
    
    

def reform_data(stks, idx):
    """
    Reshapes the data into a dataframe of stock prices at each day (index=dates, columns=stocks).
    
    :param stks: name of the stock data file; has ticker, date, and price data
                 Here taken from WRDS-accessed CRSP historical data from 1925, though none have
                 ticker names until 1962.
    :param idx: name of index file; has companies, date added, and date removed
                Also taken from WRDS-accessed CRSP historical data from index creation.
                
    The pseudocode goes as follows:
        read(stks,idx)
        stocks = list(stocks that have been in the index)
        indexlist = stks[only data for companies that are in the index]
        new_df = DataFrame(index=dates, columns = stocks)
        new_df(date,stock) = price if stock was in the index on date
        new_df(date,stock) = percent change of price from previous date
        
    :Returns:
        new_df: The new dataframe of price changes
        stockyear: First year that data is taken from (labeling purposes)
        indexname: Index being referenced (labeling purposes)
        dates: List of dates being referenced
        

    """ 
    # reading / organizing the datasets
    stockslist = pd.read_csv(stks)
    
    index = pd.read_csv(idx)
    index = index.set_index(['co_tic'])
    

    #### creating lists and relevant dataset ####
    indexlist = stockslist[stockslist['TICKER'].isin(index.index.values)] # new df only with stocks that have been in the index
    indexlist = indexlist.set_index(['TICKER', 'date'])
    indexlist = indexlist[~indexlist.index.duplicated(keep='last')] # removes duplicate values
    stocks = np.unique(indexlist.index.get_level_values(0))   # list of all stocks in index
    stocks = stocks[~pd.isnull(stocks)]
    
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
                        
                        # if there is no information for that date, instead of being 0/NA, take the previous price datum
                        if new_df.at[stock,date] == 0.0:
                            stocknum = new_df.index.get_loc(stock)
                            datenum = new_df.columns.get_loc(date)
                            new_df.iat[stocknum, datenum] = new_df.iat[stocknum, datenum-1]
        
    ### reforming new_df
    new_df = new_df.dropna(how='all')   # removes dates without any data
    new_df = new_df.T   # organize dataframe with stocks in columns (for correlation)
    new_df.sort_index(inplace=True)
    new_df = new_df.pct_change(fill_method=None)   # looks are percent change to correlate behavior rather than prices
    dates = np.unique(new_df.index.values)   # redefine dates
    new_df.to_csv('new_df.csv')
    dates_df = pd.DataFrame(dates)
    dates_df.to_csv("dates.csv", header=True, index=None)
    
    
    return new_df, dates

def correlations(dates, new_df):
    """
    Calculates the correlations for each of a predetermined time increment.
    
    :param dates: Full list of dates with stock data
    :param new_df: Full dataframe of stock data
                
    The pseudocode goes as follows:
        ranges = list of date ranges (start, end) in 60-day increments
        for each range:
            create a correlation matrix for the stock prices in this range
            add to list of correlation matrices
        
    :Returns:
        corr_list: list of all correlation matrices for the incremented dataset
        
    """ 
    corr_list = []
    ranges = list(grouper(dates, 60))

    #creates a list of correlation data for 60 day increments
    for ran in ranges:
        sel = new_df.loc[ran[0]:ran[-1]]
        corr_df = sel.corr(method='spearman')
        corr_list.append(corr_df)
        
    return corr_list

def networks(corr_list, indexname, stockyear, show_networks=False):
    """
    Creates the correlation network from correlation matrices.
    
    :param corr_list: List of correlation matrices from incremented data
    :param indexname: Name of index being referenced (labeling purposes)
    :param stockyear: Start year of data being used (labeling purposes)
    :param show_networks: Indicates whether to display the correlation networks
                
    The pseudocode goes as follows:
        for each correlation matrix:
            links = pd.DataFrame(var1, var2, value)
                links(var1,var2) = stock1, stock2
                links(value) = correlation
            links_filtered = links[correlation above predetermined threshold]
            G = graph(edges from links_filtered)
            if show_networks:
                display network
    :Returns:
        none
        
        Saves adjacency matrices to external files (for later use)
        Optionally displays the graphs using networkx
        
    """ 
    for i in range(len(corr_list)):
        # creates the filtered links for each correlation set, threshold at 0.5 somewhat arbitrarily
        corr_df = corr_list[i]
        links = corr_df.stack().reset_index()
        links.columns = ['var1', 'var2','value']
        mean = np.mean(links['value'])
        std = np.std(links['value'])
        links_filtered=links.loc[ (links['value'] > mean-std) & (links['var1'] != links['var2']) ]

        G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
        G_df = nx.to_pandas_adjacency(G)
        G_df.to_csv('Graphs/{}_{}/adjacency_matrix/range_{}.csv'.format(indexname, stockyear, i))
        
        if show_networks:
            plt.figure(i)
            nx.draw_spring(G, with_labels=True, node_color='orange', node_size=400, edge_color='black', linewidths=1, font_size=9)
    if show_networks:
        plt.show()

