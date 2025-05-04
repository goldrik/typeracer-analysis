# Aurik Sarker
# 30 April 2025

# Contains certain functions for facilitating TypeRacer parsing

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import requests
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

##
# URL HANDLING

# Return HTML text from given URL
#   Option: Save html to dictionary so to prevent repeat loading
def read_url(wp:str, htmlDict:dict=None, 
             useSelenium:bool=False, driver:webdriver.Chrome=None) -> str:
    print(wp)

    # TODO handle case where loaded HTML (in dict) was not loaded with selenium
    if htmlDict is not None:
        if wp in htmlDict:
            return htmlDict[wp]

    # Define the two ways to read the URL
    #   This is done to facilitate multiple calls (in case of timeout)
    if not useSelenium:
        def getHTML(url) -> str: return requests.get(url).text
    else:
        if driver is None:
            driver = get_selenium_driver()

        def getHTML(url) -> str:
            driver.get(url)


            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div[aria-label="A tabular representation of the data in the chart."]'))
            )

            return driver.page_source
    
    html = getHTML(wp)
    # Account for request timeout
    for s in [5, 10, 20,60]:
        # If HTML loaded successfully, we're done
        if html:
            break
        print(f'\tTimeout occurred, retrying in {s}...')
        sleep(s)
        html = getHTML(wp)

    
    str_user_invalid0 = 'We couldn\'t find a profile for username'
    str_user_invalid1 = 'There is no user'
    str_date_invalid = 'No results matching the given search criteria.'
    str_race_invalid = 'Requested data not found'
    
    # Handle exceptions
    if not html:
        raise Exception('Error: Request timed out indefinitely, exiting...')
    if (str_user_invalid0 in html) or (str_user_invalid1 in html):
        raise Exception('Error: Invalid user given, exiting...')
    if str_date_invalid in html:
        raise Exception('Error: Invalid date given, exiting...')
    if str_race_invalid in html:
        raise Exception('Error: Invalid race information given, exiting...')

    if htmlDict is not None:
        htmlDict[wp] = html

    return html


def get_selenium_driver():
    options = webdriver.chrome.options.Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    return webdriver.Chrome(options=options)



##
# DATETIME

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


##
# MISCELLANEOUS

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

    inds = list( set(indsRef) - set(indsThis) )
    return np.sort(np.array(inds))[::-1]


# Ensure dataframes dont have duplicate rows (by index) and are ordered in decending order
def adjust_dataframe_index(df:pd.DataFrame, sortDesc:bool=True) -> pd.DataFrame:
    duplicate_inds = df.index.duplicated(keep='first')
    print(f'Dropping {duplicate_inds.sum()} duplicate rows')
    df = df[~duplicate_inds].copy()

    # Dataframe should start from the most recent race
    if sortDesc:
        df.sort_index(ascending=False, inplace=True)
    
    return df
