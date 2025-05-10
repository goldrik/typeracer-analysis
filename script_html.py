# Aurik Sarker
# 23 April 2025

#%%
#!%load_ext autoreload
#!%autoreload 2

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

import os
import dotenv
import pickle
from time import time, sleep

from parse_soup import *
from typeracer_utils import *
from typing_analysis import *


#%%
## ENV VARS
dotenv.load_dotenv()
USER: str = os.getenv('USER')
FH_PKL: str = os.getenv('FH_PKL')


#%%
## VARS
# URL format strings
from url_formatstrings import url_base, url_user, url_races, url_race, url_text

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
        columns=['WPM', 'Accuracy', 'Points', 'Rank', 'Players', 'Date', 'DateTime', 'TextID', 'TypedText', 'TypingLog', ])
    for col in ['Accuracy']: races[col] = races[col].astype(float)
    for col in ['Date', 'DateTime', 'TypedText', 'TypingLog']: races[col] = races[col].astype(object)

    racers = pd.DataFrame(columns=['Racers', 'WPMs', 'Accuracies', 'TypingLogs'], dtype=object)

    typedata = pd.DataFrame(columns=['TextID', 'TypingLog', 'Sections', 'StartInds', 'WPMs', 'Mistakes'], dtype=object)
    for col in ['TextID']: typedata[col] = typedata[col].astype(int)

    texts = pd.DataFrame(columns=['Text', 'Title', 'Type', 'Author', 'Submitter', 'WPM', 'Accuracy', 'Races', 'NumWords', 'NumChars'], dtype=object)
    for col in ['WPM', 'NumWords', 'NumChars']: texts[col] = texts[col].astype(int)
    for col in ['Accuracy']: texts[col] = texts[col].astype(float)

    return races, racers, typedata, texts


def populate_races_(df, numToLoad=1e9, raceInds=None):
    # First determine the list of races 
    if raceInds is None:
        n_races = extract_num_races(USER)
        raceInds = range(n_races, 0, -1)

    racesToLoad = get_missing_indices(df, raceInds)
    racesToLoad = racesToLoad[:numToLoad]

    races_dict = {c:[] for c in df.columns if c != 'Points'}
    races_inds = []

    for race in racesToLoad:
        wp = url_race.format(USER, race)
        _, soup = read_url(wp, htmls, useSelenium=False)

        R = extract_race_details(soup)

        # TEXT ID
        textID = int( soup.find('a', string='see stats')['href'].split('?id=')[-1] )

        # We are given only the list of opponents, so add self to complete list of players
        players = [R['user']] + [r[0] for r in R['opponents']]
        races = [R['race']] + [r[1] for r in R['opponents']]

        players_users = ['Guest'] * R['racers']
        players_wpms = [-1] * R['racers']
        players_accs = [np.nan] * R['racers']
        players_tls = [''] * R['racers']
        for player_, race_ in zip(players, races):
            if player_ == R['user']:
                soup_ = soup
            else:
                _, soup_ = read_url(url_race.format(player_, race_), htmls)

            wpm, acc, rank, tl = parse_race(soup_)
            players_users[rank-1] = player_
            players_wpms[rank-1] = wpm
            players_accs[rank-1] = acc
            players_tls[rank-1] = tl

        # ONLY IF SELENIUM USED
        if False:
            mistakes, section_texts, section_wpms = extract_mistakes_sections(soup)
            
        # Verify players_wpms is monotonically decreasing, ignoring -1 values
        wpms_ = [wpm for wpm in players_wpms if wpm != -1]
        wpms_ = np.array(wpms_)
        wpms_ = wpms_[wpms_ != -1]
        np.all(wpms_[:-1] >= wpms_[1:])
        if not np.all(wpms_[:-1] >= wpms_[1:]):
            raise Exception('Extracted player WPMs are not monotonically decreasing')

        # try:
        #     dt, textID, players, players_wpms, players_accs, players_tls = \
        #         parse_race_self(soup)
        #     typingLog = players_tls[races.loc[race]['Rank'] - 1]
        #     # Use typingLog to get the actual text shown to user
        #     # This differs from the raw text found on the site
        #     TL_ = parse_typinglog_simple(typingLog)
        #     typedText = ''.join(TL_['Char'])
        # except:
        #     print(f'\t...failed to parse race {race}. Skipping')
        #     continue


        # races = pd.DataFrame(dtype=int, 
        #     columns=['DateTime', 'WPM', 'Accuracy', 'Points', 'Rank', 'Players', 
        #              'Racers, WPMs, Accuracies, TypingLogs',
        #              'TextID', 'TypedText', 'TypingLog', ])

        races_dict['DateTime'].append(R['datetime'])
        races_dict['WPM'].append(R['wpm'])
        races_dict['Accuracy'].append(R['accuracy'])
        # races_dict['Points'].append()
        races_dict['Rank'].append(R['rank'])
        races_dict['NumRacers'].append(R['racers'])
        races_dict['Racers'].append(players_users)
        races_dict['WPMs'].append(players_wpms)
        races_dict['Accuracies'].append(players_accs)
        races_dict['TypingLogs'].append(players_tls)
        races_dict['TextID'].append(textID)
        
        tl = players_tls[R['rank']-1]
        if tl != '':
            TL_ = parse_typinglog_simple(tl)
            typedText = ''.join(TL_['Char'])
        else:
            typedText = ''

        races_dict['TypedText'].append(typedText)
        races_dict['TypingLog'].append(tl)

        races_inds.append(race)
    
    if len(races_inds):
        races_ = pd.DataFrame(races_dict, index=races_inds)

        df: pd.DataFrame = pd.concat([df, races_])
        df = adjust_dataframe_index(df)

    return df


## READ HTML: RACES
def populate_races(races, numToLoad=None):
    # First, get the total number of races
    indsPrev = races.index

    n_races = extract_num_races(USER)
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
            _, soup = read_url(wp)
        except:
            print(f'\t...failed to read webpage. Stop loading races')
            break
        print(f'\t...read in {time()-starttime_:0.2f} seconds', end=' ')
        
        starttime_ = time()
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
        return url_races.format(USER, numRacesToLoad, '')
    
    # Check if this index matches the final index in the recently loaded races
    if (ind+1) == lastRaceLoaded:
        if currentPageSoup is None:
            raise Exception('ERROR: currentPageSoup must be set if lastRaceLoaded is input')
        older_div = currentPageSoup.find('a', 
                                            string=lambda text: str_older in text.lower())
        if older_div is None:
            raise Exception('ERROR: Could not find "load older results" link')
        
        return url_base + 'race_history' + older_div['href']
    
    # Otherwise, start from the date (right after) the missing race
    search_date = next_day_to_str(races.loc[ind+1, 'Date'])
    return url_races.format(USER, numRacesToLoad, search_date)


### READ HTML: RACE
def populate_racers(racers, races):
    # Check which races we have not loaded yet
    racesToLoad = get_missing_indices(racers, races)

    races_dict = {c:[] for c in ['Race', 'DateTime', 'TextID', 'TypedText', 'TypingLog']}
    racers_dict = {c:[] for c in racers.columns}

    for race in racesToLoad:
        wp = url_race.format(USER, race)
        _, soup = read_url(wp, htmls, useSelenium=False)

        try:
            dt, textID, players, players_wpms, players_accs, players_tls = \
                parse_race_self(soup)
            typingLog = players_tls[races.loc[race]['Rank'] - 1]
            # Use typingLog to get the actual text shown to user
            # This differs from the raw text found on the site
            TL_ = parse_typinglog_simple(typingLog)
            typedText = ''.join(TL_['Char'])
        except:
            print(f'\t...failed to parse race {race}. Skipping')
            continue

        races_dict['Race'].append(race)
        races_dict['DateTime'].append(dt)
        races_dict['TextID'].append(textID)
        races_dict['TypedText'].append(typedText)
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
            _, soup_ = read_url(url_race.format(player, race), htmls)

        wpm, acc, rank, tl = parse_race(soup_)
        players_users[rank-1] = player
        players_wpms[rank-1] = wpm
        players_accs[rank-1] = acc
        players_tls[rank-1] = tl


    # ONLY IF SELENIUM USED
    if False:
        mistakes, section_texts, section_wpms = extract_mistakes_sections(soup)
        
    # Verify players_wpms is monotonically decreasing, ignoring -1 values
    wpms_ = [wpm for wpm in players_wpms if wpm != -1]
    if not all(ths >= nxt for ths,nxt in zip(wpms_[:-1], wpms_[1:])):
        raise Exception('Extracted player WPMs are not monotonically decreasing')

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
                _, soup = read_url(url_text.format(textID), htmls)

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

            texts_dict['NumWords'].append( len(T['text'].split(' ')) )
            texts_dict['NumChars'].append( len(T['text']) )

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


# # if os.path.exists(FN_PKL_USER):
# if False:
#     with open(FN_PKL_USER, 'rb') as f:
#         races, racers, typedata, texts = pickle.load(f)
# else:
#     races, racers, typedata, texts = init_dataframes()
    

if os.path.exists(FN_PKL_HTMLS):
# if False:
    with open(FN_PKL_HTMLS, 'rb') as f:
        htmls = pickle.load(f)
else:
    htmls = {}


#%%
races = pd.DataFrame(dtype=int, 
    columns=['DateTime', 'WPM', 'Accuracy', 'Points', 'Rank', 'NumRacers', 
                'TextID', 'TypedText', 'TypingLog',
                'Racers', 'WPMs', 'Accuracies', 'TypingLogs', ])

races = populate_races_(races, 100)

#%%
## RACES
races = populate_races(races)
# if races.empty:
#     races = populate_races(races, 400)
# else:
#     races = populate_races(races, 10)

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

# Create dataframe which contains the three typingLog dataframes for each race

# Process typingLog
# for race in races.index:
#     textID = races.loc[race, 'TextID']

#     tl = races.loc[race, 'TypingLog']
#     text = texts.loc[textID, 'Text']

#     wpm = races.loc[race, 'WPM']
#     acc = races.loc[race, 'Accuracy']

#     TL,C,W,_ = parse_typinglog(tl, text)

#     wpm_, opt_t, opt_m = compute_wpm_best(C, wpm)
#     if np.abs(wpm_-wpm) < 0.01:
#         ye = '=='
#     else:
#         ye = 'x'
#     print(f'{race}\t{wpm:0.2f}\t{wpm_:0.2f}\t{ye}\t{opt_t:<12}\t{opt_m}')

#%%
## UPDATE

# Process typingLog
# for race in races.index:
#     textID = races.loc[race, 'TextID']

#     tl = races.loc[race, 'TypingLog']
#     text = texts.loc[textID, 'Text']

#     wpm = races.loc[race, 'WPM']
#     acc = races.loc[race, 'Accuracy']

#     TL,C,W,_ = parse_typinglog(tl, text)

#     acc_, opt_t, opt_m = compute_acc_best(C, acc)
#     if np.abs(acc_-acc) < 0.01:
#         ye = '=='
#     else:
#         ye = 'x'
#     print(f'{race}\t{acc:0.2f}\t{acc_:0.2f}\t{ye}\t{opt_t:<12}\t{opt_m}')


#%%
## SAVE
with open(FN_PKL_USER, 'wb') as f:
    pickle.dump([races, racers, typedata, texts], f)
with open(FN_PKL_HTMLS, 'wb') as f:
    pickle.dump(htmls, f)



#%%
# RACES OF NOTE

## TEXT
# Text with numbers (years) and parentheses
# ind = 7264
# With double quotes
# ind = 7498
# iive vs live
# ind = 7541

## MISTAKES
# No Mistakes
# 7506

