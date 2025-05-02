# Aurik Sarker
# 23 April 2025

#%%
import numpy as np
import pandas as pd

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

import pickle
from datetime import datetime, timedelta
from time import time, sleep
import os
import re

from ParseSoup import *


#%%
user = 'goldrik'
FH_PKL = '.'





#%%

wp_base = 'https://data.typeracer.com/pit/'
wp_user = wp_base + 'profile?user={}'
wp_races = wp_base + 'race_history?user={}&n={}&startDate={}'
wp_race = wp_base + 'result?id=|tr:{}|{}'
wp_text = wp_base + 'text_info?id={}'

str_user_invalid0 = 'We couldn\'t find a profile for username'
str_user_invalid1 = 'There is no user'
str_date_invalid = 'No results matching the given search criteria.'
str_race_invalid = 'Requested data not found'
str_older = 'load older results'

FN_PKL_USER = os.path.join(FH_PKL, f'typeracer_{user}.pkl')
FN_PKL_HTMLS = os.path.join(FH_PKL, f'typeracer_htmls.pkl')


# Set up Chrome
options = webdriver.chrome.options.Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
driver = webdriver.Chrome(options=options)


#%%
## FUNCTIONS

# Return HTML text from given URL
#   Option: Save html to dictionary so to prevent repeat loading
def read_url(wp:str, htmlDict:dict=None, useSelenium:bool=False) -> str:
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
        def getHTML(url) -> str:
            driver.get(url)

            # WAIT for the chart's aria-label div to appear
            #   This is only applicable for the race result page 
            #   (with javascript running to compute mistakes and section WPMs)
            # if 'result' in url:
            # webdriver.support.ui.WebDriverWait(driver, 15).until(
                # webdriver.support.expected_conditions.presence_of_element_located((webdriver.common.by.By.CSS_SELECTOR, 'div[aria-label="A tabular representation of the data in the chart."]'))
            # )
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


# Get number of races for given user
def get_nraces(user:str) -> int:
    html = read_url(wp_user.format(user))
    soup = BeautifulSoup(html, 'html.parser')

    stats = soup.find_all('div', class_='Profile__Stat')
    for stat in stats:
        label = stat.find('span', class_='Stat__Btm')
        if label.text.strip() == 'Races':
            value = stat.find('span', class_='Stat__Top')
            return int(value.text.strip().replace(',', ''))
        
    raise Exception('ERROR: Parsing User webpage failed')


# From a list of indices, get the largest missing indices 
#   (if the indices were sorted in reverse)
def get_next_missing_index(indices:list) -> None|int:
    if len(indices) == 0:
        return None

    indices = sorted(indices, reverse=True)
    indices.append(0)

    for i in range(len(indices) - 1):
        if (indices[i] - indices[i+1]) > 1:
            return indices[i] - 1
    
    return None


def adjust_dataframe_indices(df:pd.DataFrame, sortDesc:bool=True) -> pd.DataFrame:
    duplicate_inds = df.index.duplicated(keep='first')
    print(f'Dropping {duplicate_inds.sum()} duplicate rows')

    df = df[~duplicate_inds]

    if sortDesc:
        df.sort_index(ascending=False, inplace=True)
    
    return df


## READ HTML: RACES
def populate_races(races):
    wp = get_races_url_start(races, Nraces)

    starttime = time()
    for _ in range(3):
    # while wp != '': 
        starttime_ = time()
        try:
            html = read_url(wp)
        except:
            print(f'\t...failed to read webpage. Stop loading races')
            break
        print(f'\t...read in {time()-starttime_:0.2f} seconds', end=' ')
        
        starttime_ = time()
        soup = BeautifulSoup(html, 'html.parser')
        races_ = parse_races(soup)
        print(f'...parsed in {time()-starttime_:0.2f} seconds', end='\n')
        
        races = pd.concat([races, races_])

        wp = get_races_url_next(races, soup, races_.index.min())

    print(f'\nCompleted in {(time()-starttime)/60:0.2f} minutes')

    races = adjust_dataframe_indices(races)

    return races


def get_races_url_start(races:pd.DataFrame, totalRaces:int, numRacesToLoad:int=100) -> str:
    # Default, load the latest races
    wp_def = wp_races.format(user, numRacesToLoad, '')

    # No races loaded -> start from beginning
    if races.empty:
        return wp_def
    
    # All races loaded -> return no url
    if totalRaces == races.index.max():
        return ''
    
    # Gaps in races dataframe which may need to be filled in
    ind = get_next_missing_index(races.index)
    if ind:
        search_date = next_day_to_str(races.loc[ind+1, 'Date'])
        return wp_races.format(user, numRacesToLoad, search_date)

    raise Exception('ERROR: races DataFrame parsing failed')


def get_races_url_next(races:pd.DataFrame, currentPageSoup:BeautifulSoup, lastRaceLoaded:int, numRacesToLoad:int=100) -> str:
    ind = get_next_missing_index(races.index)
    
    # If the races matrix is complete, return no url
    if ind is None:
        return ''
    
    # Check if this index matches the final index in the recently loaded races
    if (ind+1) == lastRaceLoaded:
        older_div = currentPageSoup.find('a', string=lambda text: str_older in text.lower())
        if older_div is None:
            raise Exception('ERROR: Could not find "load older results" link')
        
        return wp_base + 'race_history' + older_div['href']

    search_date = next_day_to_str(races.loc[ind+1, 'Date'])
    return wp_races.format(user, numRacesToLoad, search_date)


### READ HTML: RACE
def populate_racers(racers, races, toLoad:int=np.inf):
    # Check which races we have not loaded yet
    #   i.e. dont have DateTime, racers, or sections
    racesToLoad = races[races['DateTime'].isna() | ~races.index.isin(racers.index)].index
    racesToLoad = racesToLoad[:min(len(racesToLoad), toLoad)]

    races_dict = {c:[] for c in ['Race', 'DateTime', 'TextID', 'TypingLog']}
    racers_dict = {c:[] for c in racers.columns}

    for race in racesToLoad:
        wp = wp_race.format(user, race)
        html = read_url(wp, htmls, useSelenium=False)
        soup = BeautifulSoup(html, 'html.parser')

        try:
            # dt, textID, typingLog, players, players_wpms, players_accs, players_tls = \
            #     parse_race(soup, races.loc[race])
            
            dt, textID, players, players_wpms, players_accs, players_tls = \
                parse_race_self(soup)
            typingLog = players_tls[races.loc[race]['Rank'] - 1]
        except:
            print(f'\t...failed to parse race {race}. Skipping')
            continue

        races_dict['Race'].append(race)
        races_dict['DateTime'].append(dt)
        races_dict['TextID'].append(textID)
        races_dict['TypingLog'].append(typingLog)

        racers_dict['Racers'].append(players)
        racers_dict['WPMs'].append(players_wpms)
        racers_dict['Accuracies'].append(players_accs)
        racers_dict['TypingLogs'].append(players_tls)


    race_inds = races_dict.pop('Race')
    if len(race_inds):
        racers_ = pd.DataFrame(racers_dict, index=race_inds)
        for c in races_dict.keys():
            for r,race in enumerate(race_inds):
                races.loc[race, c] = races_dict[c][r]

        racers = pd.concat([racers, racers_])
        racers = adjust_dataframe_indices(racers)

    return racers, races



def parse_race_self(soup:BeautifulSoup):
    # TEXT ID
    textID = int( soup.find('a', string='see stats')['href'].split('?id=')[-1] )

    # Find the race details table
    table = soup.find('table', class_='raceDetails')
    rows = table.find_all('tr')

    for row in rows:
        cols = row.find_all('td')
        col_name = cols[0].get_text(strip=True)
        col_val = cols[1].get_text(strip=True)

        if col_name == 'Racer':
            user = cols[1].find('a')['href'][13:]
        if col_name == 'Race Number':
            race = int(col_val)

        if col_name == 'Date':
            dt = str_to_datetime(col_val)

        if col_name == 'Rank':
            nRacers = int(col_val[-2])

        if col_name == 'Opponents':
            opps_links = cols[1].find_all('a')
            opps = [link.text for link in opps_links]
            opp_inds = [int(link['href'].split('|')[-1]) for link in opps_links]

    players = [user] + opps
    inds = [race] + opp_inds


    players_users = ['Guest'] * nRacers
    players_wpms = [-1] * nRacers
    players_accs = [np.nan] * nRacers
    players_tls = [''] * nRacers
    for player, race in zip(players, inds):
        if player == user:
            soup_ = soup
        else:
            html_ = read_url(wp_race.format(player, race), htmls)
            soup_ = BeautifulSoup(html_, 'html.parser')

        wpm, acc, rank, tl = parse_race(soup_)
        players_users[rank-1] = player
        players_wpms[rank-1] = wpm
        players_accs[rank-1] = acc
        players_tls[rank-1] = tl


    # ONLY IF SELENIUM USED
    if False:
        mistakes, section_texts, section_wpms = mistakes_sections_from_soup(soup)

    return dt, textID, players_users, players_wpms, players_accs, players_tls





# def parse_race(soup:BeautifulSoup, raceRow:pd.Series) -> tuple:
#     # TEXT ID
#     textID = int( soup.find('a', string='see stats')['href'].split('?id=')[-1] )
#     typingLog = soup.find('script', string=lambda st: 'typingLog' in st).text.strip()[17:-2]

#     # Find the race details table
#     table = soup.find('table', class_='raceDetails')
#     rows = table.find_all('tr')

#     links = soup.find_all('a')

#     nRacers = raceRow['Players']
#     userRank = raceRow['Rank']

#     players = ['Guest' if i+1 != userRank else user for i in range(nRacers) ]
#     players_wpms = [-1 if i+1 != userRank else int(raceRow['WPM']) for i in range(nRacers)]
#     players_accs = [np.nan if i+1 != userRank else float(raceRow['Accuracy']) for i in range(nRacers)]
#     players_tls = ['' if i+1 != userRank else typingLog for i in range(nRacers)]

#     for row in rows:
#         cols = row.find_all('td')
#         col_name = cols[0].get_text(strip=True)
#         if col_name == 'Date':
#             col_val = cols[1].get_text(strip=True)
#             dt = str_to_datetime(col_val)

#         if col_name == 'Opponents':
#             opps_links = cols[1].find_all('a')
#             opps = [link.text for link in opps_links]
#             opp_inds = [int(link['href'].split('|')[-1]) for link in opps_links]

#             rank_texts = cols[1].find_all(string=(lambda x: 'place' in x))
#             ranks = [int(text[2]) for text in rank_texts]

#             for opp, opp_race_ind, rank in zip(opps, opp_inds, ranks):
#                 wpm,acc, tl = parse_opponent_race(opp, opp_race_ind)

#                 players[rank-1] = opp
#                 players_wpms[rank-1] = wpm
#                 players_accs[rank-1] = acc
#                 players_tls[rank-1] = tl


#     # ONLY IF SELENIUM USED
#     if False:
#         mistakes, section_texts, section_wpms = mistakes_sections_from_soup(soup)

#     return dt, textID, typingLog, players, players_wpms, players_accs, players_tls



# For a particular user and race, read their race page and extract
#   average wpm, accuracy, and the typing log string
def parse_opponent_race(user:str, race:int) -> tuple:
    html = read_url(wp_race.format(user, race), htmls)
    soup = BeautifulSoup(html, 'html.parser')

    # opponents_regex = r'(\w+)\s*\((\d+)(?:\w{2}) place\)'

    # Find the race details table
    table = soup.find('table', class_='raceDetails')
    rows = table.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        col_name = cols[0].get_text(strip=True)
        if col_name == 'Speed':
            col_val = cols[1].get_text(strip=True)
            wpm = float(col_val.split(' WPM')[0])
            if wpm % 1 != 0:
                raise Exception('WPM IS NOT AN INTEGER')
            
            wpm = int(wpm)

        if col_name == 'Accuracy':
            col_val = cols[1].get_text(strip=True)
            accuracy = float(col_val[:-1])

    typingLog = extract_typing_log(soup)

    return wpm, accuracy, typingLog



# ONLY IF SELENIUM USED TO LOAD HTML
def mistakes_sections_from_soup(soup:BeautifulSoup):
    # MISTAKES
    mistake_list = soup.select('table.WordsWithErrors ol.content div.replayWord')
    mistakes = [div.get_text(strip=True) for div in mistake_list]
    # SECTIONS
    tds = [tbody.find_all('td') for tbody in soup.find_all('tbody')][-1]
    assert(len(tds) == 16)
    section_texts = [td.text.strip() for td in tds[::2]]
    section_wpms = [float(td.text.strip()) for td in tds[1::2]]

    # ! The final section_text does not seem to include the last character
    # ! Unsure if this happens always, or only when there's a period, or something

    return mistakes, section_texts, section_wpms




### READ HTML: TEXT
# For each textID in the races dataframe, 
# Populate the texts dataframe
#   avoids re-reading text if already found in texts
def populate_texts(texts:pd.DataFrame, races:pd.DataFrame) -> pd.DataFrame:
    texts_dict = {c:[] for c in texts.columns if c != 'Races'}
    textIDs = []
    for race in races.index:
        textID = races.loc[race, 'TextID']
        if np.isnan(textID): continue

        textID = int(textID)

        # Check if we've loaded this text before
        if (textID not in texts.index) and (textID not in textIDs):
            try:
                html = read_url(wp_text.format(textID), htmls)
                soup = BeautifulSoup(html, 'html.parser')

                text, title, text_type, author, submitter, wpm, acc = \
                    parse_text_soup(soup)
            except:
                print(f'\t...failed to parse text {textID}. Skipping')
                continue

            textIDs.append(textID)
            texts_dict['Text'].append(text)
            texts_dict['Title'].append(title)
            texts_dict['Type'].append(text_type)
            texts_dict['Author'].append(author)
            texts_dict['Submitter'].append(submitter)
            texts_dict['WPM'].append(wpm)
            texts_dict['Accuracy'].append(acc)

    if len(textIDs):
        texts_ = pd.DataFrame(texts_dict, index=textIDs)
        texts = pd.concat([texts, texts_])

        texts = adjust_dataframe_indices(texts, sortDesc=False)

    # Update races (for each text)
    for textID in texts.index:
        texts.at[textID, 'Races'] = races[races['TextID'] == textID].index.tolist()

    return texts


def parse_text_soup(soup:BeautifulSoup) -> tuple:
    # text_regex = r'\((\w+)\) *by (\w.*\w)\s*'
    
    text_div = soup.find('div', class_='fullTextStr')
    text_info = text_div.find_next_siblings()[0]

    wpm = int(soup.find('th', string='Avg. speed: ').find_next_sibling('td').text.strip()[:-4])
    acc = float(soup.find('th', string='Avg. accuracy:').find_next_sibling('td').text.strip()[:-1])

    text_parts = [text.strip() for text in text_info.text.split('\n')]

    text = text_div.text
    title = text_info.find('a').text
    text_type = text_parts[2][1:-1]
    author = text_parts[3][3:]

    submitter_td = soup.find('td', string='Submitted by:')
    if submitter_td is not None:
        submitter = submitter_td.find_next_sibling('td').text.strip()
    else:
        submitter = ''
    
    return text, title, text_type, author, submitter, wpm, acc

#%%
## PICKLE

# First, check for data loaded already

# htmls - dictionary url -> html
# DATAFRAMES
#   note: index = race (or textid for texts)
# races - race details
# racers - list of racers per race
#          each column 
# sections - sections per race
# texts - all races

if os.path.exists(FN_PKL_USER):
    with open(FN_PKL_USER, 'rb') as f:
        races, racers, sections, texts = pickle.load(f)
else:
    pass

races = pd.DataFrame(dtype=int, 
    columns=['WPM', 'Accuracy', 'Points', 'Rank', 'Players', 'Date', 'DateTime', 'TextID', 'TypingLog', 'Mistakes', ])
for col in ['Accuracy']: races[col] = races[col].astype(float)
for col in ['Date', 'DateTime', 'TypingLog', 'Mistakes']: races[col] = races[col].astype(object)

racers = pd.DataFrame(columns=['Racers', 'WPMs', 'Accuracies', 'TypingLogs'], dtype=object)

sections = pd.DataFrame(columns=['TextID', 'Texts', 'TextInds', 'WPMs'], dtype=object)
for col in ['TextID']: sections[col] = sections[col].astype(int)

texts = pd.DataFrame(columns=['Text', 'Title', 'Type', 'Author', 'Submitter', 'WPM', 'Accuracy', 'Races'], dtype=object)
for col in ['WPM']: texts[col] = texts[col].astype(int)
for col in ['Accuracy']: texts[col] = texts[col].astype(float)
    

if os.path.exists(FN_PKL_HTMLS):
    with open(FN_PKL_HTMLS, 'rb') as f:
        htmls = pickle.load(f)
else:
    htmls = {}


#%%
## USER
# First, get the number of races
Nraces = get_nraces(user)


#%%
## RACES
races = populate_races(races)
nRaces = len(races)

if not races.index.is_monotonic_decreasing:
    print('Warning: Races is not monotonic decreasing')
if races.index[-1] != 1:
    print('Warning: Final race read is not the first race')


#%%
## RACES
# racers, races = populate_racers(racers, races, 25)
racers, races = populate_racers(racers, races)




#%%
## TEXTS
texts = populate_texts(texts, races)


#%%
## UPDATE

# Process typingLog


#%%
## SAVE

with open(FN_PKL_USER, 'wb') as f:
    pickle.dump([races, racers, sections, texts], f)
with open(FN_PKL_HTMLS, 'wb') as f:
    pickle.dump(htmls, f)



# %%

if False:
    wp_ = 'https://data.typeracer.com/pit/text_info?id=4950000'
    html_ = requests.get(wp_).text
    soup_ = BeautifulSoup(html_, 'html.parser')
    with open('wp.html', 'w') as f:
        f.write(html_)


#%%

# 7502
# Racer 	Aurik (goldrik)
# Race Number 	7502
# Date 	Wed, 23 Apr 2025 21:37:44 +0000
# Speed 	125 WPM Try again?
# Accuracy 	97.7%
# Rank 	1st place (out of 5)
# Opponents 	danisflying (4th place) saravanangct (3rd place) 

#     few
#     times.
#     grow
#     wisdom
#     These
#     tough
#     hope
#     primitive
#     forever,
#     might
#     is
	
# Speed Throughout the Race123456780100200SegmentSpeed
# Segment	WPM
# Of course, there are those who learn after the first few times. 	150.588
# They grow out of sports. And there are others who were born with the 	134.59
# wisdom to know that nothing lasts. These are the truly tough among us, 	128.12
# the ones who can live without illusion, or without even the hope of 	140.351
# illusion. I am not that grown-up or up-to-date. I am a simpler 	114.338
# creature, tied to more primitive patterns and cycles. I need to think 	112.858
# something lasts forever, and it might as well be that state of being 	108.023
# that is a game; it might as well be that, in a green field, in the sun	135.287


# 7506
# No Mistakes

