# Aurik Sarker
# 30 April 2025

#%%
#!%load_ext autoreload
#!%autoreload 2
#!%load_ext line_profiler

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
from TypeRacerUser import TypeRacerUser
from TypingLog import TypingLog


#%%
## ENV VARS
dotenv.load_dotenv()
USER: str = os.getenv('USER')
FH_PKL: str = os.getenv('FH_PKL')

FN_PKL_USER = os.path.join(FH_PKL, f'typeracer_{USER}.pkl')
# FN_PKL_HTMLS = os.path.join(FH_PKL, f'typeracer_htmls.pkl')

try: 
    userdata
except:
    userdata = TypeRacerUser.load(FN_PKL_USER)


#%%

def print_tl_df(TL):
   [print(f'{a}\t{b}\t{c}\t{d}') for a,b,c,d in 
        zip(TL['CharInd'], TL['Op'], TL['Char'], TL['Ms'])]


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

# ind = 7511

# First race with "glitch" issue
#   character marked with keystroke from previous word
ind = 7083
# Issue with multiple deletions in one keystroke 
#   tripping up parse_typinglog() substring check (if statement)
ind = 6978

ind = 6422

# Unicode character was typed
ind = 6047

ind = 4201

ind = 2776

ind = 2753

ind = 2742

ind = 2732

ind = 2480

# Time difference and mistake char
ind = 2407
# ind = 2382

# ind = 821



#%%

inds = userdata.races.index
# inds = np.random.choice(races.index, 300, replace=False)

# inds = range(2500, 0, -1)
# inds = range(userdata.races.index.max(), 2742, -1)

# inds = [ind]
inds = [2407]

# inds = np.concatenate([np.arange(7000, 6500, -1), np.arange(1000, 0, -1)])

T_D = []
MT = []

st = time()
for race in inds:
    print(race)

    tl = userdata.races.loc[race, 'TypingLog']
    text = userdata.races.loc[race, 'TypedText']
    wpm = userdata.races.loc[race, 'WPM']
    acc = userdata.races.loc[race, 'Accuracy']

    ####################

    T = TypingLog(tl)

    T.generate()
    T.parse_chars()

    TL0 = T._chars
    TL = T.entries
    C = T.chars
    W = T.words
    text_ = T.text

    ####################

    assert(tl.count('|') == 1)
    assert(T._tl[1][0] == '0')

    # assert(len(tp)-1 == TL['Stroke'].iloc[-1])
    assert(''.join(C['Char']) == ''.join(TL0['Char']))
    assert(text_ == text)

    # TODO Add a check to see if 4 or more entries at the end are identical

    
    assert(''.join(C['Char']) == text)
    assert(''.join(W['Word']) == text)
    
    assert(TL0['Ms'].sum() == TL['Ms'].sum())
    assert(TL['Ms'].sum() == C['Ms'].sum())
    assert(TL['Ms'].sum() == W['Ms'].sum())

    nw = len(text.split(' '))
    assert(nw == C['Word'].iloc[-1]+1)
    assert(nw == len(W))

    # for wa,WA in zip([wpm,acc], ['wpm', 'acc']):
    #     wa_, opt_t, opt_m = compute_wpm_acc_best(C, wa, WA)
    #     print(f'\t{wa:0.2f}\t{wa_:0.2f}\t{opt_t:<12}\t{opt_m}')

    t_d = np.count_nonzero(C['Ms'] - TL0['Ms'])
    if t_d:
        T_D.append([race, t_d])
        print("\tTime Difference between TL0 and C:", t_d)
        print('\t', np.where(C['Ms'] - TL0['Ms'])[0])

    # For characters marked as typed CORRECTLY, check Char == Typed
    assert((C['Mistake'] | (C['Char'] == C['Typed'])).all())
    # How many characters were marked MISTAKE, but Char == Typed
    C_ = (C['Mistake'] & (C['Char'] == C['Typed']))
    mt = np.count_nonzero(C_)
    if mt:
        MT.append([race, mt])
        print("\tMistake Char has correct Typed Char:", mt)
        print('\t', np.where(C_)[0])
        if (C[C_].Char == '.').all():
            print('\tAll were periods')


print(f'\t{ (time()-st):0.2f} secs')
print(len(T_D), len(MT))



#%%
# Recreate keystrokes and windows
inds = userdata.races.index
# inds = np.concatenate([np.arange(7000, 6500, -1), np.arange(1000, 0, -1)])
# inds = [7511]

for race in inds:
    print(race)
    tl = userdata.races.loc[race, 'TypingLog']

    T = TypingLog(tl)
    T.generate()
    T.parse_chars()

    tl_header = tl[:[ind for ind, c in enumerate(tl) if c == ','][2]]
    
    TL0 = T._chars
    TL = T.entries
    C = T.chars
    W = T.words

    S = T.strokes
    WW = T.windows
    text_inds = WW['TextInd']
    num_strokes = WW['NumStrokes']

    strokes = []
    stroke_ms = []
    window_num = []
    windows = []

    for w, Ww in TL.groupby('Window'):
        window = ''

        for _, Ss in Ww.groupby('Stroke'):
            ms = Ss['Ms'].sum()

            stroke = ''
            for r in Ss.itertuples():
                entry = f'{r.WindowInd}{r.Op}{r.Char}'
                stroke += entry

            strokes.append(stroke)
            stroke_ms.append(ms)
            window_num.append(w)

            window += f'{ms},{stroke},'

        windows.append(window[:-1])

    S_ = pd.DataFrame({
        'Stroke': strokes,
        'Ms': stroke_ms,
        'Window': window_num,
    })
    W_ = pd.DataFrame({
        'Window': windows,
        'NumStrokes': num_strokes,
        'TextInd': text_inds,
    })


    W__ = pd.DataFrame({
        'TextInd': W_['TextInd'],
        'NumStrokes': W_['NumStrokes'],
        'Window': W_['Window'],
    })
    tl1_ = ','.join(W__.astype(str).values.flatten()) + ','
    tl0_ = ''
    # for r in TL0.itertuples():
    #     c = r.Char
    #     if c.isdigit() or c == '-':
    #         c = '\\b' + str(c)
    #     tl0_ += c + str(r.Ms)

    def convertC(c):
        if c.isdigit() or c == '-':
            return '\\b' + c 
        else:
            return TypingLog.reverse_char_clean(c)
    tl0_ = ''.join(TL0['Char'].apply(convertC) + TL0['Ms'].apply(str))
    
    # tl0_ = ''.join([TypingLog.reverse_char_clean(c) for c in tl0_])
    tl1_ = ''.join([TypingLog.reverse_char_clean(c) for c in tl1_])

    tl_ = tl_header + ',' + tl0_ + '|' + tl1_
    
    assert(S.equals(S_))
    assert(WW.equals(W_))
    # assert(tl1 == tl1_)
    assert(tl == tl_)

