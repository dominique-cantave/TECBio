
from processing import *
import sys
import pandas as pd

indexname = 'DJ'
stockyear = 1925

dates = pd.read_csv('dates.csv')
new_df = pd.read_csv('new_df.csv', index_col=0)

correlations = correlations(dates, new_df)

networks(correlations, indexname, stockyear, show_networks=False)