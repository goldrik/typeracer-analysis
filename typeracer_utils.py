# Aurik Sarker
# 30 April 2025

# Contains certain functions for facilitating TypeRacer parsing

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Convert string (from webpage) to datetime
def str_to_datetime(col_str:str):
    # Special case: Sept
    col_str = col_str.replace('Sept', 'Sep')

    if col_str.strip().lower() == "today":
        return datetime.today().date()
    if '+' in col_str:
        return datetime.strptime(col_str, '%a, %d %b %Y %H:%M:%S %z').date()
    if '.' in col_str:
        return datetime.strptime(col_str, '%b. %d, %Y').date()
    else:
        return datetime.strptime(col_str, '%B %d, %Y').date()

# Takes datetime (from DataFrame) and converts to date string for webpage
#   This is done to retrieve races missing from DataFrame, i.e. races preceding the given date
#   increment the date so that the webpage returns races from the current day as well (some may have been missed)
def next_day_to_str(dt) -> str:
    return (dt + timedelta(days=1)).strftime('%Y-%m-%d')


# Given two iterables (list, array), find the indices in indsRef which are missing from indsThis
#   If an integer ind2 is given instead (for indsRef), use a list of indices from 1 to N
#   If a dataframe is given, just take its indices
#   If indsRef is not given, just find the missing indices within indsThis, alone
def get_missing_indices(indsThis, indsRef=None) -> np.ndarray:
    if isinstance(indsThis, pd.DataFrame):
        indsThis = indsThis.index

    if indsRef is None:
        indsRef = np.max(indsThis)

    if isinstance(indsRef, pd.DataFrame):
        indsRef = indsRef.index
    elif not hasattr(indsRef, '__iter__'):
        indsRef = range(1,indsRef+1)

    return np.array(list( set(indsRef) - set(indsThis) ))


# Ensure dataframes dont have duplicate rows (by index) and are ordered in decending order
def adjust_dataframe_index(df:pd.DataFrame, sortDesc:bool=True) -> pd.DataFrame:
    duplicate_inds = df.index.duplicated(keep='first')
    print(f'Dropping {duplicate_inds.sum()} duplicate rows')
    df = df[~duplicate_inds]

    # Dataframe should start from the most recent race
    if sortDesc:
        df.sort_index(ascending=False, inplace=True)
    
    return df
