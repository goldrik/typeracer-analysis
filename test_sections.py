# Aurik Sarker
# 04 May 2025

# Test if our computation of text sections matches TypeRacer's

#!%load_ext autoreload
#!%autoreload 2

import numpy as np
import pandas as pd

from time import time

from bs4 import BeautifulSoup
from typeracer_utils import read_url, get_selenium_driver
from parse_soup import extract_num_races, extract_mistakes_sections, extract_typing_log
from typing_analysis import text_to_sections, parse_typinglog, compute_wpm_best

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
races = np.random.choice(inds, 25, replace=False)

# races = [7264]
# races = [5305]

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
        mistakes, section_texts, section_wpms = extract_mistakes_sections(soup)

        tl = extract_typing_log(soup)
        try:
            TL,C,W,_ = parse_typinglog(tl, text)
        except Exception as e:
            printf('Error in parsing typingLog')
            printf('\t', e)
            continue
        
        printf('===== MISTAKES: TYPERACER VS COMPUTED =====')

        mistakes_ = W[W['Mistake']]['Word'].to_list()
        mistakes_ = [m.rstrip() for m in mistakes_]
        if (len(mistakes_) != len(mistakes)) or (not all([m_==m for m_,m in zip(mistakes_,mistakes)])):
            printf(f'NO One or more mistakes ({len(mistakes)}) does not match')
            printf('\t', ' '.join(mistakes))
            printf('\t', ' '.join(mistakes_))
        else:
            printf(f'YE All mistakes ({len(mistakes)}) match')


        printf('===== SECTION WPM: TYPERACER VS COMPUTED =====')
        
        word_ind = 0
        for section,wpm in zip(section_texts, section_wpms): 
            nwords = len(section.split(' '))
            inds = range(word_ind, word_ind+nwords)

            W_ = W.loc[inds]
            C_ = C[C['WordInd'].apply(lambda i: i in inds)]
            
            wpm_, opt_t, opt_m = compute_wpm_best(C_, wpm)

            # section_ = ''.join(W_['Word'])[:-1]
            section_ = ''.join(C_['Char'])[:-1]
            if section_ != section:
                printf('NO TypeRacer section text does not match words DataFrame')
                printf('\t', section)
                printf('\t', section_)

            if wpm == wpm_:
                printf('YE', end=' ')
            else:
                printf('NO', end=' ')
            # printf(f'\t{wpm:0.2f}\t{wpm_:0.2f}')
            printf(f'\t{wpm:0.2f}\t{wpm_:0.2f}\t{opt_t}\t{opt_m}')
            word_ind += nwords


        printf('===== SECTIONS: TYPERACER VS TEXT =====')
        if ' '.join(section_texts) == text:
            printf('YE Section text matches exactly')
        else:
            section_texts_ = section_texts.copy()
            section_texts_[-1] += text[-1]

            if ' '.join(section_texts_) == text:
                printf(f'YE Section text matches after appending last character "{text[-1]}" to section')
            else:
                printf('NO Section text does not match even after appending last cahracter')


        printf('===== SECTIONS: TYPERACER VS COMPUTED =====')
        section_texts_ = text_to_sections(text)

        # _ = [(print(s0), print(s1)) for s0,s1 in zip(section_texts, section_texts_)]
        matches = [section0 == section1 for section0, section1 in zip(section_texts, section_texts_)]
        num_matches = sum(matches)
        if all(matches):
            printf('YE', end=' ')
        else:
            printf('NO', end=' ')
        printf(f'{num_matches}/{len(section_texts)} sections match')
        if not all(matches):
            for section0, section1 in zip(section_texts, section_texts_):
                printf(f'\t\t{section0}\n\t\t{section1}')
        if not all(matches):
            # Check if the last section was the only incorrect one (the rest of the sections were fine)
            if all(matches[:-1]):
                if np.abs(len(section_texts[-1]) - len(section_texts_[-1])) == 1:
                    printf(f'\tLast section is only off by one character')

        printf('')
        printf('')


