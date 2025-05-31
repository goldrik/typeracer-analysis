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

# NOTE: Exceptions are to indicate unexpected TypeRacer webpage format only


import numpy as np
import pandas as pd

import re

from bs4 import BeautifulSoup
from typeracer_utils import str_to_datetime


##
# https://data.typeracer.com/pit/profile ...
def parse_profile_info(soup:BeautifulSoup) -> int:
    info = {}

    name_parts = soup.find_all('h2')[1].text.strip().split('\n')
    if len(name_parts) == 3:
        info['Name'] = name_parts[0]
    if len(name_parts) not in [1,3]:
        raise Exception('ERROR: Unexpected "name (user)" format in TypeRacer profile page')

    im_str = soup.find('div', class_='profileCar')['style']
    # avatarname.ext (2-4 chars, to account for some file extensions)
    avatar_match = re.search(r'avatars/(.*)\.(.{2,4})"\)', im_str)
    if avatar_match:
        info['Avatar'] = avatar_match.group(1)
        info['AvatarFilename'] = avatar_match.group(1) + '.' + avatar_match.group(2)
    else:
        raise Exception('ERROR: Parsing User avatar failed')

    spans = soup.find('div', class_='About').find_all('span')
    assert(len(spans) % 2 == 0)
    for i in range(0, len(spans), 2):
        label = spans[i].text[:-1]
        value = spans[i+1].text
        if label == 'Racing Since':
            value = str_to_datetime(value)
        elif label == 'Location':
            locs = [loc.strip() for loc in value.split('\n')]
            locs = [loc for loc in locs if loc != '']
            value = ' '.join(locs)
        elif label == 'Keyboard':
            pass
        else:
            raise Exception('ERROR: Unexpected section found in User profile page')
        
        info[label] = value

    # It is valid for these values to be missing from TypeRacer profile
    for label in ['Name', 'Location', 'Keyboard']:
        if label not in info:
            info[label] = ''

    return info


# https://data.typeracer.com/pit/profile ...
def parse_profile_stats(soup:BeautifulSoup) -> int:
    stats = {}

    stats_div = soup.find_all('div', class_='Profile__Stat')
    for stat in stats_div:
        label = stat.find('span', class_='Stat__Btm').text.strip()
        value = stat.find('span', class_='Stat__Top').text.strip()
        if label == 'Full Avg.':
            value = float(value.split(' WPM')[0])
        elif label == 'Best Race':
            value = int(value.split(' WPM')[0])
        elif label == 'Races':
            value = int(value.replace(',', ''))
        elif label == 'WPM %':
            value = float(value.replace('%', ''))
        elif label == 'Skill Level':
            pass
        elif label == 'Exp Level':
            pass
        else:
            raise Exception('ERROR: Unexpected section found in User profile page')
        
        stats[label] = value
    return stats


##
# Return DataFrame with basic info for each race found in the table
# https://data.typeracer.com/pit/race_history ...
def parse_results(soup:BeautifulSoup) -> pd.DataFrame:
    # Extract data rows from the 'Scores__Table__Row' divs
    rows_html = soup.find_all('div', class_='Scores__Table__Row')

    # Cols to save into the DataFrame
    rows_df = ['Speed', 'Accuracy', 'Points', 'Place', 'NumRacers', 'Date']
    rows = {c:[] for c in ['Race'] + rows_df}
    for row_html in rows_html:
        row_text = [col.get_text(strip=True) for col in row_html.find_all('div')]
        rows['Race'].append(int(row_text[0]))
        rows['Speed'].append(int(row_text[1][:-4]))
        rows['Accuracy'].append(float(row_text[2][:-1]))
        rows['Points'].append(int(row_text[3]))
        rows['Place'].append(int(row_text[4][0]))
        rows['NumRacers'].append(int(row_text[4][-1]))
        rows['Date'].append(str_to_datetime(row_text[5]).date())

    return pd.DataFrame( {k:rows[k] for k in rows_df}, index=rows['Race'])


##
# Return single-race results for a user
# In this project, it is used to supplement parse_race_self(), 
#   extracting minimal information for the opponents' races

# Parses the raceDetails table found in the race page
# Returns dictionary with each details from each row
def parse_race(soup:BeautifulSoup) -> tuple[dict, dict]:
    table = soup.find('table', class_='raceDetails')
    rows = table.find_all('tr')

    # Initialize opponents lists in case none exist
    opps = []
    opp_races = []
    opp_ranks= []

    details = {}
    # List of names and race numbers
    opponents = {}
    for row in rows:
        cols = row.find_all('td')
        col_name = cols[0].get_text(strip=True)
        col_val = cols[1].get_text(strip=True)

        if col_name == 'Racer':
            user_link = cols[1].find('a')['href']
            user_str = '?user='
            details[col_name] = user_link[ user_link.find(user_str) + len(user_str) : ]

        elif col_name == 'Race Number':
            details[col_name] = int(col_val)

        elif col_name == 'Date':
            details[col_name] = str_to_datetime(col_val)

        elif col_name == 'Speed':
            wpm = float(col_val.split(' WPM')[0])
            if wpm % 1 != 0:
                raise Exception(f'WPM {wpm:.2f} IN WEBPAGE WAS NOT AN INTEGER')
            details[col_name] = int(wpm)

        elif col_name == 'Accuracy':
            details[col_name] = float(col_val[:-1])

        elif col_name == 'Rank':
            details[col_name] = int(col_val[0])
            details['NumRacers'] = int(col_val[-2])

        elif col_name == 'Opponents':
            opps_links = cols[1].find_all('a')
            opps = [link.text for link in opps_links]
            opp_races = [int(link['href'].split('|')[-1]) for link in opps_links]
            opp_ranks= re.findall(r'\((\d)\w{2} place\)', col_val)

        else:
            raise Exception('ERROR: Unexpected entry found in race details table')
        
    # Save opponents separately
    opponents['Users'] = opps
    opponents['Races'] = opp_races
    opponents['Ranks'] = opp_ranks

    return details, opponents


# This is separate from the race details table
def extract_textid(soup:BeautifulSoup) -> str:
    return int( soup.find('a', string='see stats')['href'].split('?id=')[-1] )


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
def extract_mistakes_sections(soup:BeautifulSoup):
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

    text_parts = [text.strip() for text in text_info.text.split('\n') if text.strip()]

    # NOTE: Text in the raw HTML may include double-spaces (for some reason)
    # Eliminate these (and any other possible multi-spaces)
    text_ = text_div.text
    text_ = text_.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text_ = ' '.join([w for w in text_.split(' ') if w != ''])
    text_details['text'] = text_

    
    try:
        # text_details['title'] = text_info.find('a').text.strip()
        # Update: 30 May 2025
        #   Typeracer update to text info pages, removed text hyperlink
        text_details['title'] = text_parts[-3]
    except:
        text_details['title'] = ''
    
    text_details['type'] = text_parts[-2][1:-1]
    text_details['author'] = text_parts[-1][3:]

    submitter_td = soup.find('td', string='Submitted by:')
    if submitter_td is not None:
        text_details['submitter'] = submitter_td.find_next_sibling('td').text.strip()
    else:
        text_details['submitter'] = ''
    
    return text_details

