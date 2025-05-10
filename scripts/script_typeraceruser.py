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

FN_PKL_HTMLS = os.path.join(FH_PKL, f'typeracer_htmls.pkl')

#%%
## OBJECT
from TypeRacerUser import TypeRacerUser

user = TypeRacerUser('goldrik', fn_htmls=FN_PKL_HTMLS)
# user.update()
user.update(10)


#%%
# user.populate_races(10)

user.populate_results()
user.results
