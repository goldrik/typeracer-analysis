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


#%%
## ENV VARS
dotenv.load_dotenv()
USER: str = os.getenv('USER')
FH_PKL: str = os.getenv('FH_PKL')

FN_PKL_USER = os.path.join(FH_PKL, f'typeracer_{USER}.pkl')
FN_PKL_HTMLS = os.path.join(FH_PKL, f'typeracer_htmls.pkl')

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

ind = 2407
# ind = 2382

ind = 821


#%%

inds = userdata.races.index
# inds = np.random.choice(races.index, 300, replace=False)
# inds = range(2500, 0, -1)
# inds = range(races.index.max(), 2742, -1)
# inds = [ind]
# inds = [6047]
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

    assert(tl.count('|') == 1)
    assert(tl[typinglog_pipe(tl)+1] == '0')

    ####################

    TL0 = parse_typinglog_simple(tl)
    # TL = parse_typinglog_complete(tl)[0]
    # text_, tp = reconstruct_text_typinglog(TL)


    # lc, fo = [], []
    # for _,W in TL.groupby('Window'):
    #     lc.append(W.iloc[-1]['Char'])
    #     fo.append(W.iloc[0]['Op'])
    # lc = np.array(lc)
    # fo = np.array(fo)
    # if ~np.all(lc[:-1] == ' '):
    #     print('\tThere was a window that didnt end with space')
    # if ~np.all(fo == '+'):
    #     print('\tThere was a window that didnt start with an addition')

    # a = time()
    TL, C, W, text_  = \
        parse_typinglog(tl)
    # print(f'\t{ (time()-a)*1e3:0.2f} ms')



    # assert(len(tp)-1 == TL['Stroke'].iloc[-1])
    assert(text_ == text)

    
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

    C_ = C[~C['Mistake']]
    assert((C_['Char'] == C_['Typed']).all())
    C_ = C[C['Mistake']]
    mt = (C_['Char'] == C_['Typed']).sum()
    if mt:
        MT.append([race, mt])
        print("\tMistake Char has correct Typed Char:", mt)


print(f'\t{ (time()-st):0.2f} secs')
print(len(T_D), len(MT))


#%%
# Recreate keystrokes and windows
inds = userdata.races.index
# inds = np.concatenate([np.arange(7000, 6500, -1), np.arange(1000, 0, -1)])
# inds = [6047]

for race in inds:
    print(race)
    tl = userdata.races.loc[race, 'TypingLog']
    
    ind = typinglog_pipe(tl)
    tl1 = tl[ind+1:]
    
    TL, S, W = parse_typinglog_complete(tl)
    text_inds, num_strokes = parse_typinglog_wordvals(tl)

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
    tl1_ = ''.join([reverse_char_clean(c) for c in tl1_])
    
    assert(S.equals(S_))
    assert(W.equals(W_))
    assert(tl1 == tl1_)

