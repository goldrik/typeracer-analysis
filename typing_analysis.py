# Aurik Sarker
# 3 May 2025

# Parse and analyse the typingLog data
#   javascript variable extracted from the race webpage
# typingLog parsing may differ from TypeRacer's javascript methods

# Includes other functions for parsing the text

import numpy as np
import pandas as pd
import re

##
# TEXT

# Divide text into roughtly equal-length sections
# This may not match the sections identified by TypeRacer (found on each race webpage)
def text_to_sections(text: str, numSections: int=8) -> list[str]:
    section_len_chars = len(text) / numSections
    # Exact characters where the text may be divided into sections
    inds = (np.arange(numSections+1) * section_len_chars).astype(int)

    ind_end = 0
    sections = []
    for i in range(numSections):
        # ind_start = inds[i]
        # ind_end = inds[i+1]

        ind_start = ind_end
        ind_end= ind_start + int((len(text)-ind_start) / (numSections-i))

        # print(text[ind_end])

        # If the end_index is not at a space and not at the end of the text, adjust
        if ind_end < len(text) and not text[ind_end].isspace():
            ind_before = text.rfind(' ', 0, ind_end)
            ind_after = text.find(' ', ind_end)

            ss_len = lambda ind: abs(section_len_chars - (ind - ind_start))
            if ss_len(ind_before) < ss_len(ind_after):
                ind_end = ind_before
            else:
                ind_end = ind_after

        inds[i+1] = ind_end
        sections.append(text[ind_start:ind_end].strip())

    sections[-1] = sections[-1][:-1]

    return sections


##
# STATS

# Takes the characters DataFrame (computed by parse_typinglog()) and computes WPM
# Ideal for computing section WPM by inputting subsets the DataFrame
# Allows for computing WPM without spaces, etc
def compute_wpm_acc(df:pd.DataFrame, text_opt=None, ms_opt=None) -> float:
    # If only one option is given, use for both
    if ms_opt is None:
        ms_opt = text_opt

    ms = df['Ms'].to_numpy().copy()
    mistakes = df['Mistake'].to_numpy().copy()
    text_ = df['Char'].to_numpy().copy()

    text = text_
    if text_opt == 'no_spaces':
        text = text_[text_ != ' ']
    elif text_opt == 'no_endchar':
        text = text_[:-1]
    elif text_opt == 'no_endspace':
        if text_[-1] == ' ':
            text = text_[:-1]
    T = len(text)

    if ms_opt == 'no_spaces':
        ms = ms[text_ != ' ']
        mistakes = mistakes[text_ != ' ']
    elif ms_opt == 'no_endchar':
        ms = ms[:-1]
        mistakes = mistakes[:-1]
    elif ms_opt == 'no_endspace':
        if text_[-1] == ' ':
            ms = ms[:-1]
            mistakes = mistakes[:-1]
    MS = ms.sum()
    nc = np.count_nonzero(~mistakes)

    wpm = (T/5) / (MS/1e3/60)
    acc = (nc / T) * 100
    return wpm, acc


# FOR TESTING PURPOSES
# Loop through all (purportedly) ways to compute WPM/Accuracy
# Return the value with minimum error and its options
def compute_wpm_acc_best(df, target, wpm_or_acc):
    opts = ['all', 'no_spaces', 'no_endchar', 'no_endspace']

    if wpm_or_acc == 'wpm':
        output_ind = 0
    elif wpm_or_acc == 'acc':
        output_ind = 1

    min_diff = np.inf
    estimate = np.inf
    opt_t, opt_m = 'all', 'all'
    for m in opts:
        for t in opts:
            estimate_ = compute_wpm_acc(df, t,m)[output_ind]
            diff = abs(target - estimate_)
            if diff < min_diff:
                min_diff = diff
                estimate = estimate_
                opt_t, opt_m = t, m

    return estimate, opt_t, opt_m


