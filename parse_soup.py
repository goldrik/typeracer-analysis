# Aurik Sarker
# 30 April 2025

# Contains functions for parsing TypeRacer pages
# The pages must already have been read in (to html string) and procesed using BeautifulSoup
#   i.e. BeautifulSoup(html, 'html.parser')

# The inputs are all BeautifulSoup objects
#   https://data.typeracer.com/pit/profile?user={}
#   https://data.typeracer.com/pit/race_history?user={}&n={}&startDate={}
#   https://data.typeracer.com/pit/result?id=|tr:{}|{}
#   https://data.typeracer.com/pit/text_info?id={}
# The outputs may be
#   DataFrame - for profile page (table of races)
#   tuple - for pages corresponding to one single race or text

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from typeracer_utils import str_to_datetime

##
# Return number of races
# https://data.typeracer.com/pit/profile ...
def extract_num_races(soup:BeautifulSoup) -> int:
    stats = soup.find_all('div', class_='Profile__Stat')
    for stat in stats:
        label = stat.find('span', class_='Stat__Btm')
        if label.text.strip() == 'Races':
            value = stat.find('span', class_='Stat__Top')
            return int(value.text.strip().replace(',', ''))
        
    raise Exception('ERROR: Parsing User webpage failed')


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
# In this project, it is used to supplement parse_race_self(), 
#   extracting minimal information for the opponents' races
def parse_race(soup:BeautifulSoup) -> tuple[int, float, int, str]:
    R = extract_race_details(soup)
    typing_log = extract_typing_log(soup)

    return R['wpm'], R['accuracy'], R['rank'], typing_log


# Parses the raceDetails table found in the race page
# Returns dictionary with each details from each row
def extract_race_details(soup:BeautifulSoup) -> dict:
    table = soup.find('table', class_='raceDetails')
    rows = table.find_all('tr')

    # Initialize opponents lists in case none exist
    opps = []
    opp_races = []

    race_details = {}
    for row in rows:
        cols = row.find_all('td')
        col_name = cols[0].get_text(strip=True)
        col_val = cols[1].get_text(strip=True)

        if col_name == 'Racer':
            race_details['user'] = cols[1].find('a')['href'][13:]

        elif col_name == 'Race Number':
            race_details['race'] = int(col_val)

        elif col_name == 'Date':
            race_details['datetime'] = str_to_datetime(col_val)

        elif col_name == 'Speed':
            wpm = float(col_val.split(' WPM')[0])
            if wpm % 1 != 0:
                raise Exception(f'WPM {wpm:.2f} IN WEBPAGE WAS NOT AN INTEGER')
            race_details['wpm'] = int(wpm)

        elif col_name == 'Accuracy':
            race_details['accuracy'] = float(col_val[:-1])

        elif col_name == 'Rank':
            race_details['rank'] = int(col_val[0])
            race_details['racers'] = int(col_val[-2])

        elif col_name == 'Opponents':
            opps_links = cols[1].find_all('a')
            opps = [link.text for link in opps_links]
            opp_races = [int(link['href'].split('|')[-1]) for link in opps_links]

        else:
            pass

    # Each entry in "opponents" is [opponent name, their race number]
    race_details['opponents'] = [[opp,race] for opp,race in zip(opps,opp_races)]

    return race_details


# Function to handle cases where typingLog js variable doesnt exist
def extract_typing_log(soup:BeautifulSoup) -> str:
    try:
        return soup.find('script', string=lambda st: 'typingLog' in st).text.strip()[17:-2]
    except: 
        # Some users dont have a typing log for some reason
        return ''


# Function to load list of mistakes and sections (w/ WPMs)
# ONLY IF SELENIUM USED TO LOAD HTML
#   relies on typeracer to run internal javascript to compute these values
def mistakes_sections_from_soup(soup:BeautifulSoup):
    # MISTAKES
    mistake_list = soup.select('table.WordsWithErrors ol.content div.replayWord')
    # if len(mistake_list) == 0:
    #     raise Exception('ERROR: Mistakes not found in webpage.' + 
    #                     '\n\tTyperacer javascript may not have run or output mistakes')

    mistakes = [div.get_text(strip=True) for div in mistake_list]

    # SECTIONS
    tds = [tbody.find_all('td') for tbody in soup.find_all('tbody')][-1]
    if len(tds) != 16:
        raise Exception('ERROR: Parsing sections failed')
    
    section_texts = [td.text.strip() for td in tds[::2]]
    section_wpms = [float(td.text.strip()) for td in tds[1::2]]

    # ! The final section_text does not seem to include the last character
    # ! Unsure if this happens always, or only when there's a period, or something

    return mistakes, section_texts, section_wpms


##
# Parses the text page for details, returns dictionary
# https://data.typeracer.com/pit/text_info ...
def parse_text(soup:BeautifulSoup) -> dict:
    text_details = {}

    # text_regex = r'\((\w+)\) *by (\w.*\w)\s*'
    
    text_div = soup.find('div', class_='fullTextStr')
    text_info = text_div.find_next_siblings()[0]

    text_details['wpm'] = \
        int(soup.find('th', string='Avg. speed: ').find_next_sibling('td').text.strip()[:-4])
    text_details['accuracy'] = \
        float(soup.find('th', string='Avg. accuracy:').find_next_sibling('td').text.strip()[:-1])

    text_parts = [text.strip() for text in text_info.text.split('\n')]

    # NOTE: Text in the raw HTML may include double-spaces (for some reason)
    # Eliminate these (and any other possible multi-spaces)
    text_ = text_div.text
    text_ = text_.replace('\r', ' ').replace('\n', ' ')
    text_ = ' '.join([w for w in text_.split(' ') if w != ''])
    text_details['text'] = text_

    text_details['title'] = text_info.find('a').text
    text_details['type'] = text_parts[2][1:-1]
    text_details['author'] = text_parts[3][3:]

    submitter_td = soup.find('td', string='Submitted by:')
    if submitter_td is not None:
        text_details['submitter'] = submitter_td.find_next_sibling('td').text.strip()
    else:
        text_details['submitter'] = ''
    
    return text_details

