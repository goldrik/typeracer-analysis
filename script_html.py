# Aurik Sarker
# 23 April 2025

#%%
#!%load_ext autoreload
#!%autoreload 2

from bs4.element import NavigableString, PageElement
import numpy as np
import pandas as pd

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup, Tag

import pickle
from time import time, sleep
import os

from parse_soup import *
from typeracer_utils import *


#%%
USER = 'goldrik'
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

FN_PKL_USER = os.path.join(FH_PKL, f'typeracer_{USER}.pkl')
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



###
def init_dataframes():
    races = pd.DataFrame(dtype=int, 
        columns=['WPM', 'Accuracy', 'Points', 'Rank', 'Players', 'Date', 'DateTime', 'TextID', 'TypingLog', ])
    for col in ['Accuracy']: races[col] = races[col].astype(float)
    for col in ['Date', 'DateTime', 'TypingLog']: races[col] = races[col].astype(object)

    racers = pd.DataFrame(columns=['Racers', 'WPMs', 'Accuracies', 'TypingLogs'], dtype=object)

    sections = pd.DataFrame(columns=['TextID', 'TypingLog', 'Sections', 'StartInds', 'WPMs', 'Mistakes'], dtype=object)
    for col in ['TextID']: sections[col] = sections[col].astype(int)

    texts = pd.DataFrame(columns=['Text', 'Title', 'Type', 'Author', 'Submitter', 'WPM', 'Accuracy', 'Races'], dtype=object)
    for col in ['WPM']: texts[col] = texts[col].astype(int)
    for col in ['Accuracy']: texts[col] = texts[col].astype(float)

    return races, racers, sections, texts


## READ HTML: RACES
# Note, this may load slighty more races than given by numToLoad
def populate_races(races, numToLoad=None):
    # First, get the total number of races
    html = read_url(wp_user.format(USER))
    soup = BeautifulSoup(html, 'html.parser')

    n_races = extract_num_races(soup)

    if numToLoad is None:
        numToLoad = n_races
    racesToLoad = range(n_races, n_races-numToLoad, -1)

    wp = get_next_races_url(races, racesToLoad)

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

        wp = get_next_races_url(races, racesToLoad, lastRaceLoaded=races_.index.min(), currentPageSoup=soup)

    print(f'\nCompleted in {(time()-starttime)/60:0.2f} minutes')

    races = adjust_dataframe_index(races)
    return races


# Given the list of loaded races and list of races to load
#   return the URL to load the next set of missing races
# lastRaceLoaded and currentPageSoup are used handle the "older results" link
#   both or neither variable must be set at once
def get_next_races_url(races:pd.DataFrame, raceInds:int, numRacesToLoad:int=100, 
                       lastRaceLoaded:int=None, currentPageSoup:BeautifulSoup=None) -> str:
    # Gaps in races dataframe which may need to be filled in
    inds = get_missing_indices(races, raceInds)
    if not inds.size:
        # All races loaded -> return no url
        return ''
    
    ind = inds.max()
    # If the latest race is missing, start from the beginning
    if ind == np.max(raceInds):
        return wp_races.format(USER, numRacesToLoad, '')
    
    # Check if this index matches the final index in the recently loaded races
    if (ind+1) == lastRaceLoaded:
        if currentPageSoup is None:
            raise Exception('ERROR: currentPageSoup must be set if lastRaceLoaded is input')
        older_div = currentPageSoup.find('a', 
                                            string=lambda text: str_older in text.lower())
        if older_div is None:
            raise Exception('ERROR: Could not find "load older results" link')
        
        return wp_base + 'race_history' + older_div['href']
    
    # Otherwise, start from the date (right after) the missing race
    search_date = next_day_to_str(races.loc[ind+1, 'Date'])
    return wp_races.format(USER, numRacesToLoad, search_date)


# def get_races_url_start(races:pd.DataFrame, totalRaces:int, numRacesToLoad:int=100) -> str:
#     # Gaps in races dataframe which may need to be filled in
#     inds = get_missing_indices(races, totalRaces)
#     if not inds.size:
#         # All races loaded -> return no url
#         return ''
    
#     ind = inds.max()
#     # If the latest race is missing, start from the beginning
#     if ind == totalRaces:
#         return wp_races.format(USER, numRacesToLoad, '')
    
#     # Otherwise, start from the date (right after) the missing race
#     search_date = next_day_to_str(races.loc[ind+1, 'Date'])
#     return wp_races.format(USER, numRacesToLoad, search_date)


# def get_races_url_next(races:pd.DataFrame, currentPageSoup:BeautifulSoup, lastRaceLoaded:int, numRacesToLoad:int=100) -> str:
#     inds = get_missing_indices(races)
#     # Races dataframe is complete -> return no url
#     if not inds.size:
#         return ''
    
#     ind = inds.max()
#     # Check if this index matches the final index in the recently loaded races
#     if (ind+1) == lastRaceLoaded:
#         older_div = currentPageSoup.find('a', string=lambda text: str_older in text.lower())
#         if older_div is None:
#             raise Exception('ERROR: Could not find "load older results" link')
        
#         return wp_base + 'race_history' + older_div['href']

#     # Otherwise, take the next missing race and start from its date (techically the day after)
#     search_date = next_day_to_str(races.loc[ind+1, 'Date'])
#     return wp_races.format(USER, numRacesToLoad, search_date)



### READ HTML: RACE
def populate_racers(racers, races):
    # Check which races we have not loaded yet
    racesToLoad = get_missing_indices(racers, races)

    races_dict = {c:[] for c in ['Race', 'DateTime', 'TextID', 'TypingLog']}
    racers_dict = {c:[] for c in racers.columns}

    for race in racesToLoad:
        wp = wp_race.format(USER, race)
        html = read_url(wp, htmls, useSelenium=False)
        soup = BeautifulSoup(html, 'html.parser')

        try:
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
        racers = adjust_dataframe_index(racers)

    races['TextID'] = races['TextID'].astype(int)
    return racers, races



# Return infrmation used to populate the typeracer dataframes
#   For the user, this includes the text id, precise datetime, and opponent info
# Opponent information is parsed using parse_soup.parse_race()
# https://data.typeracer.com/pit/result ...
def parse_race_self(soup:BeautifulSoup):
    # TEXT ID
    textID = int( soup.find('a', string='see stats')['href'].split('?id=')[-1] )

    R = extract_race_details(soup)

    # We are given only the list of opponents, so add self to complete list of players
    players = [R['user']] + [r[0] for r in R['opponents']]
    races = [R['race']] + [r[1] for r in R['opponents']]

    players_users = ['Guest'] * R['racers']
    players_wpms = [-1] * R['racers']
    players_accs = [np.nan] * R['racers']
    players_tls = [''] * R['racers']
    for player, race in zip(players, races):
        if player == R['user']:
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

    return R['datetime'], textID, players_users, players_wpms, players_accs, players_tls



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

                T = parse_text(soup)
            except:
                print(f'\t...failed to parse text {textID}. Skipping')
                continue

            textIDs.append(textID)
            texts_dict['Text'].append(T['text'])
            texts_dict['Title'].append(T['title'])
            texts_dict['Type'].append(T['type'])
            texts_dict['Author'].append(T['author'])
            texts_dict['Submitter'].append(T['submitter'])
            texts_dict['WPM'].append(T['wpm'])
            texts_dict['Accuracy'].append(T['accuracy'])

    if len(textIDs):
        texts_ = pd.DataFrame(texts_dict, index=textIDs)
        texts = pd.concat([texts, texts_])

        texts = adjust_dataframe_index(texts, sortDesc=False)

    # Update races (for each text)
    for textID in texts.index:
        texts.at[textID, 'Races'] = races[races['TextID'] == textID].index.tolist()

    return texts



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

# if os.path.exists(FN_PKL_USER):
if False:
    with open(FN_PKL_USER, 'rb') as f:
        races, racers, sections, texts = pickle.load(f)
else:
    races, racers, sections, texts = init_dataframes()
    

if os.path.exists(FN_PKL_HTMLS):
    with open(FN_PKL_HTMLS, 'rb') as f:
        htmls = pickle.load(f)
else:
    htmls = {}



#%%
## RACES
# races = populate_races(races)
races = populate_races(races, 185)

if not races.index.is_monotonic_decreasing:
    print('Warning: Races is not monotonic decreasing')
if races.index[-1] != 1:
    print('Warning: Final race read is not the first race')


#%%
## RACES
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



#%%


# a = read_url(wp_race.format(USER, 7506), useSelenium=True)
# selenium HTTPConnectionPool(host='localhost', port=52298): Read timed out. (read timeout=120)


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

