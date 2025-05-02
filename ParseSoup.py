# Aurik Sarker
# 30 April 2025

# Contains modules for parsing TypeRacer pages
# The pages must already have been read in (to html string) and procesed using BeautifulSoup
#   i.e. BeautifulSoup(html, 'html.parser')

# The inputs are all BeautifulSoup objects
# The outputs may be
#   DataFrame - for profile page (table of races)
#   tuple - for pages corresponding to one single race or text

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from bs4 import BeautifulSoup
# TODO change the file structure
# from script_html import str_to_datetime

##
# Return DataFrame with basic info for each race found in the table
# https://data.typeracer.com/pit/race_history ...
def parse_races(soup:BeautifulSoup) -> pd.DataFrame:
    # Extract data rows from the 'Scores__Table__Row' divs
    rows_html = soup.find_all('div', class_='Scores__Table__Row')

    # Cols to save into the DataFrame
    rows_df = ['WPM', 'Accuracy', 'Points', 'Rank', 'Players', 'Date']
    rows = {c:[] for c in ['Race'] + rows_df}
    for row_html in rows_html:
        row_text = [col.get_text(strip=True) for col in row_html.find_all('div')]
        rows['Race'].append(int(row_text[0]))
        rows['WPM'].append(int(row_text[1][:-4]))
        rows['Accuracy'].append(float(row_text[2][:-1]))
        rows['Points'].append(int(row_text[3]))
        rows['Rank'].append(int(row_text[4][0]))
        rows['Players'].append(int(row_text[4][-1]))
        rows['Date'].append(str_to_datetime(row_text[5]))

    return pd.DataFrame( {k:rows[k] for k in rows_df}, index=rows['Race'])


##
# Return single-race results for a user
# https://data.typeracer.com/pit/result ...
def parse_race(soup:BeautifulSoup):
    # Find the race details table
    table = soup.find('table', class_='raceDetails')
    rows = table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        col_name = cols[0].get_text(strip=True)
        col_val = cols[1].get_text(strip=True)

        if col_name == 'Speed':
            wpm = float(col_val.split(' WPM')[0])
            if wpm % 1 != 0:
                raise Exception('WPM IS NOT AN INTEGER')
            
            wpm = int(wpm)

        if col_name == 'Accuracy':
            accuracy = float(col_val[:-1])

        if col_name == 'Rank':
            rank = int(col_val[0])

    typingLog = extract_typing_log(soup)

    return wpm, accuracy, rank, typingLog


def extract_typing_log(soup:BeautifulSoup):
    try:
        return soup.find('script', string=lambda st: 'typingLog' in st).text.strip()[17:-2]
    except: 
        # Some users dont have a typing log for some reason
        return ''









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