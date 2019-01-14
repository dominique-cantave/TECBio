#
#
# Author: Dominique Cantave
# Project: TECBio 2018, University of Pittsburgh
# Mentors: Anne-Ruxandra Carvunis, David Koes
# stock_process.py -> transforms stock data into useable form
#
#

"""
This script calls the processing functions from processing.py.
"""
from processing import *
import sys


# Takes stock data and index data from sys.argv inputs
stockset = sys.argv[1]
stockyear = stockset.split(os.path.sep)[-1].split('_')[2][0:4]
indexset = sys.argv[2]
indexname = indexset.split(os.path.sep)[-1].split('_')[0]

# transforms data
new_df, dates = reform_data(stockset, indexset)

# creates correlation matrices
correlations = correlations(dates, new_df)

# creates network graphs
networks(correlations, indexname, stockyear, show_networks=False)



