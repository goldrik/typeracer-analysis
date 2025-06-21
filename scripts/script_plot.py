# Aurik Sarker
# 21 June 2025

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
from url_formatstrings import *

from TypeRacerUser import TypeRacerUser
from TypingLog import TypingLog
from Keyboard import Keyboard


#%%
## ENV VARS
dotenv.load_dotenv()
USER: str = os.getenv('USER')
FH_PKL: str = os.getenv('FH_PKL')

FN_PKL_USER = os.path.join(FH_PKL, f'typeracer_{USER}.pkl')
FN_PKL_HTMLS = os.path.join(FH_PKL, 'typeracer_htmls.pkl')


#%%
## LOAD
userdata = TypeRacerUser.load(FN_PKL_USER)

# Generate TypingLog output for each race
TLs=  {}
for race in userdata.races.index:
    TLs[race] = TypingLog(userdata.races.at[race, 'TypingLog'])
    TLs[race].generate()


#%%
dts = userdata.races['DateTime']
times = dts.apply(lambda x: x.time().minute)
times = dts.apply(lambda x: x.date().day)

# plt.plot(dts, times, '.')

# plt.hist(times, bins=range(24))
# plt.hist(times)
# plt.plot(dts, userdata.races['Accuracy'], '.')

wpm = userdata.races['WPM'].to_numpy()[::-1]
wpm_avg = [round(np.mean(wpm[i:i+100])) for i in range(0, len(wpm), 100)]
plt.plot(wpm_avg, '.')


