# Aurik Sarker
# 30 April 2025

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
# from typing_analysis import *
from typing_analysis_ import *


#%%
## ENV VARS
dotenv.load_dotenv()
USER: str = os.getenv('USER')
FH_PKL: str = os.getenv('FH_PKL')

FN_PKL_USER = os.path.join(FH_PKL, f'typeracer_{USER}.pkl')
FN_PKL_HTMLS = os.path.join(FH_PKL, f'typeracer_htmls.pkl')

try: 
    races
except:
    with open(FN_PKL_USER, 'rb') as f:
        races, racers, typedata, texts = pickle.load(f)
    with open(FN_PKL_HTMLS, 'rb') as f:
        htmls = pickle.load(f)


#%%

def print_tl_df(TL):
   [print(f'{a}\t{b}\t{c}\t{d}') for a,b,c,d in 
        zip(TL['CharInd'], TL['Op'], TL['Char'], TL['Ms'])]


#%%
# Text with numbers (years) and parentheses
ind = 7264
# With double quotes
# ind = 7498

# iive vs live
# ind = 7541

# ind = 7511

# First race with "glitch" issue
#   character marked with keystroke from previous word
ind = 7083
# Issue with multiple deletions in one keystroke 
#   tripping up parse_typinglog() substring check (if statement)
ind = 6978

ind = 6422

ind = 4201

ind = 2776

ind = 2753

ind = 2742

ind = 2732


#%%

inds = races.index
# inds = range(2163-1, 0, -1)
# inds = range(races.index.max(), 2742, -1)
# inds = np.random.choice(races.index, 300, replace=False)
# inds = [ind]
# inds = [7548]

for race in inds:
    print(race)

    textID = races.loc[race, 'TextID']

    tl = races.loc[race, 'TypingLog']
    text = texts.loc[textID, 'Text']
    wpm = races.loc[race, 'WPM']
    acc = races.loc[race, 'Accuracy']

    # chars_total = texts.loc[textID, 'NumChars']
    # words_total = texts.loc[textID, 'NumWords']

    ####################

    assert(tl.count('|') == 1)
    assert(tl[typinglog_pipe(tl)+1] == '0')

    ####################

    TL0 = parse_typinglog_simple(tl)
    # assert(''.join(TL0['Char']) == text)
    # if ''.join(TL0['Char']) != text:
    #     print(textID)
    #     print(text)
    #     print(''.join(TL0['Char']))
    #     continue

    ####################

    TL = parse_typinglog_complete(tl)[0]

    lc, fo = [], []
    for _,W in TL.groupby('Window'):
        lc.append(W.iloc[-1]['Char'])
        fo.append(W.iloc[0]['Op'])
    lc = np.array(lc)
    fo = np.array(fo)

    # if ~np.all(lc[:-1] == ' '):
    #     print('\tThere was a window that didnt end with space')
    if ~np.all(fo == '+'):
        print('\tThere was a window that didnt start with an addition')

    text_ = reconstruct_text_typinglog(TL)
    assert(text_ == text)


#%%

raise Exception

inds = races.index
# inds = range(2800, 0, -1)
# inds = np.random.choice(races.index, 300, replace=False)
# inds = [ind]
# inds = [7538]

for race in inds:
    # if race in [2751, 2742]:
    #     continue
    print(race)

    textID = races.loc[race, 'TextID']

    tl = races.loc[race, 'TypingLog']
    text = texts.loc[textID, 'Text']
    wpm = races.loc[race, 'WPM']
    acc = races.loc[race, 'Accuracy']

    chars_total = texts.loc[textID, 'NumChars']
    words_total = texts.loc[textID, 'NumWords']

    # chars_total = len(text)
    # words_total = len(text.split(' '))

    assert(tl.count('|') == 1)
    assert(tl[typinglog_pipe(tl)+1] == '0')

    TL0 = parse_typinglog_simple(tl)
    assert(''.join(TL0['Char']) == text)


    # TL1, _ = tl1_to_char_ms(tl, text)
    TL1 = parse_typinglog_complete(tl)
    TL1, C,W, text_ = parse_typinglog(tl, text)
    _, word_strokes = parse_typinglog_wordvals(tl)


    # print_tl_df(TL1)
    

    wpm0 = (chars_total / 5) / (TL0['Ms'].sum() / 1e3 / 60)
    wpm1 = (chars_total / 5) / (TL1['Ms'].sum() / 1e3 / 60)
    assert(wpm0 == wpm1)

    # print(f'{wpm} {wpm1:.2f}')

    assert(TL1['WordInd'].max()+1 == words_total)
    assert(sum(word_strokes) == TL1.index[-1]+1)
    
    assert(text_ == text)
    assert(''.join(C['Char']) == text)
    assert(''.join(W['Word']) == text)

    assert(TL1['Ms'].sum() == C['Ms'].sum())
    assert(TL1['Ms'].sum() == W['Ms'].sum()) 

    # acc_ = 100 * (1 - C['Mistake'].sum() / len(C))
    # inds_chars = (C['Char'] != ' ').to_numpy()
    # acc__ = 100 * (1 - C.loc[inds_chars, 'Mistake'].sum() / np.count_nonzero(inds_chars))
    # print(f'{acc:.2f}\t{acc_:.2f}\t{acc__:.2f}')

    # assert(all(np.array(word_strokes) == W['Keystrokes'].to_numpy()))
    if len(word_strokes) != len(W) or \
        not all(np.array(word_strokes) == W['Keystrokes'].to_numpy()):
        print('\tWord Strokes Issue')

    mistake_words = W['Mistake'].to_numpy()
    assert(all(W.loc[~mistake_words, 'Word'] == W.loc[~mistake_words, 'Attempt']))


    a = []
    ops = TL1['Op'].to_numpy()
    charinds = TL1['CharInd'].to_numpy()
    for i in range(len(TL1)-1):
        if (ops[i] == '+') and (ops[i+1] == '+'):
            if charinds[i+1] < charinds[i]:
                a.append(charinds[i+1]) 

    assert(not any(a))
