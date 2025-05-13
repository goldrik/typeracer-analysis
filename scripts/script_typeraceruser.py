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
from url_formatstrings import *
from TypeRacerUser import TypeRacerUser


#%%
# Test loading TypeRacerUser totally new (don't save)
fresh = True

## ENV VARS
dotenv.load_dotenv()
USER: str = os.getenv('USER')
FH_PKL: str = os.getenv('FH_PKL')

FN_PKL_USER = os.path.join(FH_PKL, f'typeracer_{USER}.pkl')
if fresh:
    FN_PKL_HTMLS = None
else:
    FN_PKL_HTMLS = os.path.join(FH_PKL, f'typeracer_htmls.pkl')

#%%
## LOAD
try: 
    userdata = TypeRacerUser.load(FN_PKL_USER)
except:
    userdata = TypeRacerUser('goldrik', fn_htmls=FN_PKL_HTMLS)


#%%
## LOAD
# Make sure it has the htmls loaded
userdata.load_htmls()
# userdata.update()
# userdata.update(1)

# userdata.populate_results()


#%%
## SAVE
if not fresh:
    userdata.save(FN_PKL_USER)
