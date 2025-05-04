# Aurik Sarker
# 23 April 2025

#%%
#!%load_ext autoreload
#!%autoreload 2

from bs4.element import NavigableString, PageElement
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup, Tag

import os
import dotenv
import pickle
from time import time, sleep

from parse_soup import *
from typeracer_utils import *


#%%
## ENV VARS
dotenv.load_dotenv()
USER: str = os.getenv('USER')
FH_PKL: str = os.getenv('FH_PKL')


#%%
## VARS
# URL format strings
wp_base = 'https://data.typeracer.com/pit/'
wp_user = wp_base + 'profile?user={}'
wp_races = wp_base + 'race_history?user={}&n={}&startDate={}'
wp_race = wp_base + 'result?id=|tr:{}|{}'
wp_text = wp_base + 'text_info?id={}'

FN_PKL_USER = os.path.join(FH_PKL, f'typeracer_{USER}.pkl')
FN_PKL_HTMLS = os.path.join(FH_PKL, f'typeracer_htmls.pkl')


# Set up Chrome
selenium_driver = get_selenium_driver()
# pd.options.mode.copy_on_write = True

#%%
## FUNCTIONS

###
def init_dataframes():
    races = pd.DataFrame(dtype=int, 
        columns=['WPM', 'Accuracy', 'Points', 'Rank', 'Players', 'Date', 'DateTime', 'TextID', 'TypingLog', ])
    for col in ['Accuracy']: races[col] = races[col].astype(float)
    for col in ['Date', 'DateTime', 'TypingLog']: races[col] = races[col].astype(object)

    racers = pd.DataFrame(columns=['Racers', 'WPMs', 'Accuracies', 'TypingLogs'], dtype=object)

    typedata = pd.DataFrame(columns=['TextID', 'TypingLog', 'Sections', 'StartInds', 'WPMs', 'Mistakes'], dtype=object)
    for col in ['TextID']: typedata[col] = typedata[col].astype(int)

    texts = pd.DataFrame(columns=['Text', 'Title', 'Type', 'Author', 'Submitter', 'WPM', 'Accuracy', 'Races'], dtype=object)
    for col in ['WPM']: texts[col] = texts[col].astype(int)
    for col in ['Accuracy']: texts[col] = texts[col].astype(float)

    return races, racers, typedata, texts


## READ HTML: RACES
def populate_races(races, numToLoad=None):
    # First, get the total number of races
    html = read_url(wp_user.format(USER))
    soup = BeautifulSoup(html, 'html.parser')

    indsPrev = races.index

    n_races = extract_num_races(soup)
    racesMissing = get_missing_indices(races, n_races)

    if numToLoad is None:
        numToLoad = n_races
    racesToLoad = racesMissing[:numToLoad]

    # This is what the dataframe indices should be afterwards
    indsAfter = np.array(list(set(indsPrev) | set(racesToLoad)))

    wp = get_next_races_url(races, racesToLoad, n_races)

    starttime = time()
    while wp != '': 
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

        wp = get_next_races_url(races, racesToLoad, n_races, lastRaceLoaded=races_.index.min(), currentPageSoup=soup)

    print(f'\nCompleted in {(time()-starttime)/60:0.2f} minutes')

    races = races.loc[indsAfter].copy()
    races = adjust_dataframe_index(races)
    return races


# Given the list of loaded races and list of races to load
#   return the URL to load the next set of missing races
# lastRaceLoaded and currentPageSoup are used handle the "older results" link
#   both or neither variable must be set at once
def get_next_races_url(races:pd.DataFrame, raceInds:int, maxRaces:int, numRacesToLoad:int=100, 
                       lastRaceLoaded:int=None, currentPageSoup:BeautifulSoup=None) -> str:
    str_older = 'load older results'

    # Gaps in races dataframe which may need to be filled in
    inds = get_missing_indices(races, raceInds)
    if not inds.size:
        # All races loaded -> return no url
        return ''
    
    ind = inds[0]
    # If the latest race is missing, start from the beginning
    if ind == maxRaces:
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
# races - basic race details
# racers - list of racers per race
#          certain columns are lists 
#            (e.g list of opponents, their wpms, etc)
# typedata - typing data per race
#            includes sections, mistakes, etc
# texts - all texts found in given races


# if os.path.exists(FN_PKL_USER):
if False:
    with open(FN_PKL_USER, 'rb') as f:
        races, racers, typedata, texts = pickle.load(f)
else:
    races, racers, typedata, texts = init_dataframes()
    

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
    pickle.dump([races, racers, typedata, texts], f)
with open(FN_PKL_HTMLS, 'wb') as f:
    pickle.dump(htmls, f)



#%%


# a = read_url(wp_race.format(USER, 7506), useSelenium=True, driver=selenium_driver)
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

