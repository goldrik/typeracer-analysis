# Aurik Sarker
# 04 May 2025

# Test if our computation of text sections matches TypeRacer's

import numpy as np
import pandas as pd

from time import time

from bs4 import BeautifulSoup
from typeracer_utils import read_url, get_selenium_driver
from parse_soup import extract_num_races, mistakes_sections_from_soup
from typing_analysis import text_to_sections

from url_formatstrings import url_user, url_race

#%%
## INIT

# Write out log
FN_LOG = './test_sections.txt'

# For user Goldrik, get number of races
user = 'goldrik'
_, soup = read_url(url_user.format(user))
n_races = extract_num_races(soup)

inds = np.arange(1, n_races+1)
races = np.random.choice(inds, 2, replace=False)

# races = [7264]

selenium_driver = get_selenium_driver()


#%%
## TEST

with open(FN_LOG, 'a') as f:
    printf = lambda *args, **kwargs: print(*args, **kwargs, file=f)
    for race in races:
        url = url_race.format('goldrik', race)
        printf('================================================================================')
        printf(f'Reading user {user}, race {race}')
        printf(url)

        st = time()
        try:
            _, soup = read_url(url, useSelenium=True, driver=selenium_driver)
            printf(f'\tSelenium read in {time()-st:0.2f} secs.')
        except: 
            printf(f'\tSelenium failed to read ({time()-st:0.2f} secs). Skipping.')
            continue

        text = soup.find('div', class_='fullTextStr').text
        mistakes, section_texts, section_wpms = mistakes_sections_from_soup(soup)

        printf('===== SECTIONS FROM TYPERACER =====')
        if ' '.join(section_texts) == text:
            printf('YE Section text matches exactly')
        else:
            section_texts_ = section_texts.copy()
            section_texts_[-1] += text[-1]

            if ' '.join(section_texts_) == text:
                printf(f'YE Section text matches after appending last character "{text[-1]}" to section')
            else:
                printf('NO Section text does not match even after appending last cahracter')


        printf('===== COMPUTED SECTIONS =====')
        section_texts_ = text_to_sections(text)

        # _ = [(print(s0), print(s1)) for s0,s1 in zip(section_texts, section_texts_)]
        matches = [section0 == section1 for section0, section1 in zip(section_texts, section_texts_)]
        num_matches = sum(matches)
        if all(matches):
            printf('YE', end=' ')
        else:
            printf('NO', end=' ')
        printf(f'{num_matches}/{len(section_texts)} sections match')
        for section0, section1 in zip(section_texts, section_texts_):
            printf(f'\t\t{section0}\n\t\t{section1}')
        if not all(matches):
            # Check if the last section was the only incorrect one (the rest of the sections were fine)
            if all(matches[:-1]):
                if np.abs(len(section_texts[-1]) - len(section_texts_[-1])) == 1:
                    printf(f'\tLast section is only off by one character')

        printf('')
        printf('')


