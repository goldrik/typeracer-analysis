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
from bs4 import BeautifulSoup

from url_formatstrings import url_user

##
# URL HANDLING

# Return HTML text and BeautifulSoup object from given URL
#   Option: Save html to dictionary so to prevent repeat loading
def read_url(wp:str, htmlDict:dict=None, reloadHtml=False, 
             useSelenium:bool=False, driver:webdriver.Chrome=None) -> str:
    # print(wp)

    # TODO handle case where loaded HTML (in dict) was not loaded with selenium
    if not reloadHtml and htmlDict is not None:
        if wp in htmlDict:
            html = htmlDict[wp]
            soup = BeautifulSoup(html, 'html.parser')
            
            return html, soup

    # Define the two ways to read the URL
    #   This is done to facilitate multiple calls (in case of timeout)
    if not useSelenium:
        def getHTML(url) -> str: return requests.get(url).text
    else:
        if driver is None:
            driver = get_selenium_driver()

        def getHTML(url) -> str:
            driver.get(url)

            WebDriverWait(driver, 20).until(
                # EC.presence_of_element_located((By.CSS_SELECTOR, 'div[aria-label="A tabular representation of the data in the chart."]'))
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div[aria-label="A chart."]'))
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

    
    str_error_429 = '429 Too Many Requests'
    str_user_invalid0 = 'We couldn\'t find a profile for username'
    str_user_invalid1 = 'There is no user'
    str_date_invalid = 'No results matching the given search criteria.'
    str_race_invalid = 'Requested data not found'
    str_text_invalid = 'Text not found'
    
    # Handle exceptions
    if not html:
        raise Exception('Error: Request timed out indefinitely, exiting...')
    if str_error_429 in html:
        raise Exception('Error: 429 - Too many requests, exiting...')
    if (str_user_invalid0 in html) or (str_user_invalid1 in html):
        raise Exception('Error: Invalid user given, exiting...')
    if str_date_invalid in html:
        raise Exception('Error: Invalid date given, exiting...')
    if str_race_invalid in html:
        raise Exception('Error: Invalid race information given, exiting...')
    if str_text_invalid in html:
        raise Exception('Error: Invalid text ID given, exiting...')

    if htmlDict is not None:
        htmlDict[wp] = html

    soup = BeautifulSoup(html, 'html.parser')
    return html, soup


def get_selenium_driver():
    options = webdriver.chrome.options.Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    return webdriver.Chrome(options=options)



##
# DATETIME

# Convert string (from webpage) to datetime
def str_to_datetime(date_str:str):
    # Special case: Sept
    date_str = date_str.replace('Sept', 'Sep')

    # if date_str.strip().lower() == "today":
    #     return datetime.today()
    # if '+' in date_str:
    #     return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
    # if '.' in date_str:
    #     return datetime.strptime(date_str, '%b. %d, %Y')
    # else:
    #     return datetime.strptime(date_str, '%B %d, %Y')
    
    # 05 May 2025: Typeracer changed their date formats for some reason?
    if date_str.strip().lower() == "today":
        return datetime.today()
    if ':' in date_str:
        return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S')
    else:
        return datetime.strptime(date_str, '%b %d, %Y')


# Takes datetime (from DataFrame) and converts to date string for webpage
#   This is done to retrieve races missing from DataFrame, i.e. races preceding the given date
#   increment the date so that the webpage returns races from the current day as well (some may have been missed)
def next_day_to_str(dt) -> str:
    return (dt + timedelta(days=1)).strftime('%Y-%m-%d')


##
# MISCELLANEOUS

# Return number of races
# https://data.typeracer.com/pit/profile ...
def extract_num_races(user:str) -> int:
    _, soup = read_url(url_user.format(user))
    stats = soup.find_all('div', class_='Profile__Stat')
    for stat in stats:
        label = stat.find('span', class_='Stat__Btm')
        if label.text.strip() == 'Races':
            value = stat.find('span', class_='Stat__Top')
            return int(value.text.strip().replace(',', ''))
        
    raise Exception('ERROR: Parsing User webpage failed')


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

