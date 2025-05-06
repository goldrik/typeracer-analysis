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

# Determine the sections
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
# TYPINGLOG

# The typingLog contains two formats, separated by a pipe character
# To account for the possiblity of a pipe in the text itself, search for a specific regex pattern
#   i.e. pipe character followed by digits (first word index, should always be 0)
def typinglog_pipe(tl: str) -> int:
    pipe_pattern = r'\|\d+,'
    pipe_occurence = list(re.finditer(pipe_pattern, tl))
    ind = pipe_occurence[-1].start()

    return pipe_occurence[-1].start()


# Outputs three dataframes: all keystrokes, text characters, words
def parse_typinglog(tl, text):
    # This gives us all the keystrokes/entries first
    TL = parse_typinglog_complete(tl)
    num_entries = len(TL)

    words=  text.split(' ')
    words[:-1] = [w+' ' for w in words[:-1]]

    ## CHARACTERS

    # For each character in the text
    chars_text = []
    word_inds = []
    # Time it took to type each character CORRECTLY
    # i.e. includes time for incorrect keystrokes
    chars_ms = []
    # Was there a mistake made when typing this character
    mistakes = []
    # Which character did the user type
    chars_typed = []

    def reset_word_vars():
        # char_ms, mistake, first_typed_char
        return 0, False, ''


    text_ = ''
    word_ = []
    word_ind = 0
    char_ind = 0
    char_ms, mistake, first_typed_char = reset_word_vars()
    for i in range(num_entries):
        c = TL.iloc[i]['Char']
        char_ind_ = TL.iloc[i]['CharInd']
        op = TL.iloc[i]['Op']

        if op == '+':
            word_.append(c)
            assert(word_[char_ind_] == c)

        elif op == '-':
            assert(word_[char_ind_] == c)
            word_.pop(char_ind_)

        elif op == '$':
            word_[char_ind_] = c
            assert(word_[char_ind_] == c)

        # Accumulate the time needed to type this character
        char_ms += TL.iloc[i]['Ms']

        # For each character in the word, use the word's substring (so far)
        #   check if the user has typed the substring correctly yet
        substring = words[word_ind][:char_ind+1]
        if ''.join(word_) == substring:
            # The correct character
            c_ = words[word_ind][char_ind]
            chars_text.append(c_)
            word_inds.append(word_ind)

            chars_ms.append(char_ms)
            mistakes.append(mistake)
            if not mistake: 
                # Normally, the correct character was typed
                chars_typed.append(c_)
            else:
                chars_typed.append(first_typed_char)

            char_ind += 1
            char_ms, mistake, first_typed_char = reset_word_vars()
        else:
            mistake = True
            # Set this variable for just the first character
            if first_typed_char == '':
                first_typed_char = c

        # If we've reached the end of the text or end of word
        if (i == num_entries-1) or \
            ((c == ' ') and (TL.iloc[i+1]['CharInd'] == 0) and (TL.iloc[i+1]['Op'] == '+')):
            text_ += ''.join(word_)
            word_ = []

            word_ind += 1
            char_ind = 0
            char_ms, mistake, first_typed_char = reset_word_vars()

    C = pd.DataFrame({'WordInd':word_inds, 'Char':chars_text, 'Ms':chars_ms, 'Mistake':mistakes, 'Typed':chars_typed})

    ## WORDS

    words_ms = []
    # The full attempt for this word
    word_keystrokes = []
    word_mistakes = []
    word_lens = []
    # Number of keystrokes needed (not entries)
    word_strokes = []

    word_inds_0 = TL['WordInd'].to_numpy()
    word_inds_1 = C['WordInd'].to_numpy()

    ops = TL['Op'].to_numpy()
    inds_add = (ops == '+') | (ops == '$')
    
    assert(len(np.unique(word_inds)) == len(words))
    for i in range(len(words)):
        inds_0 = word_inds_0 == i
        inds_1 = word_inds_1 == i
        words_ms.append(TL['Ms'][inds_0].sum())

        word_keystrokes.append(''.join(TL['Char'][inds_0 & inds_add]))
        word_mistakes.append(C['Mistake'][inds_1].any())

        word_lens.append(len(words[i]))

        inds = TL[inds_0].index
        word_strokes.append(inds.max() - inds.min() + 1)

    W = pd.DataFrame({'Word':words, 'Attempt':word_keystrokes, 'Ms':words_ms, 'Mistake':word_mistakes, 'Length':word_lens, 'Keystrokes':word_strokes})

    return TL, C, W, text_


# Parse the second half of typingLog
#   This contains *all* the keystrokes
def parse_typinglog_complete(tl:str):
    # Split by |
    T = tl[typinglog_pipe(tl)+1:]
    # Do this to make regex better
    T = ',' + T[:-1]

    # Split into words
    wordnums_pattern = r',(\d+),(\d+),'
    # This makes it so regex matches are all unique (no overlaps)
    wordnums_pattern_ = wordnums_pattern.replace('(', '').replace(')', '')
    
    # Each word now comprises of multiple keystrokes
    # Format for each entry: 
    #   ,{ms},{ind}{op}{char},
    ms_pattern = r',-?\d+,'
    char_pattern = r'(\d+)([+\-$])(\\"|.)'

    word_attempts = re.split(wordnums_pattern_, T)[1:]

    # Arrays for DataFrame, values for each entry
    times = []
    char_inds = []
    ops = []
    chars = []

    # Which word
    word_inds = []
    # DataFrame index is for each typingLog keystroke (can have duplicates)
    #   ** a keystroke may have multiple "entries" (i.e. selection -> replace)
    pd_inds = []

    # NOTE: 
    # Text has multiple words -> the user attempts the word -> the attempt is comprised of multiple keystrokes 
    #   -> an keystroke is usually just one typingLog entry, but can be multiple (if select & replace)

    i = -1
    # Loop through each full word (i.e. all keystrokes for that attempt)
    for w,W in enumerate(word_attempts):
        W_ = ',' + W
        ms_vals = re.findall(ms_pattern, W_)
        keystrokes = re.split(ms_pattern, W_)[1:]

        for ms_str, keystroke in zip(ms_vals, keystrokes):
            ms = int(ms_str[1:-1])

            entries = re.findall(char_pattern, keystroke)
            # Iterate the typingLog keystroke (dataframe index)
            i += len(entries) != 0
            assert(len(entries) > 0)

            for char_ind, op, char in entries:
                times.append(ms)

                char_inds.append(int(char_ind))
                ops.append(op)
                
                # The "character" found with regex can be \" (two characters), so take the last index
                chars.append(char[-1])

                pd_inds.append(i)
                word_inds.append(w)

                # Only save the ms once (to avoid double-dipping)
                ms = 0

    TL = pd.DataFrame({'WordInd':word_inds, 'CharInd':char_inds, 'Char':chars, 'Ms':times, 'Op':ops}, index=pd_inds)
    return TL


# Parse the first half of typingLog
# This contains all the correct characters and the time it took to *correctly* type each
#   * not all keystrokes, only correct ones shown
def parse_typinglog_simple(tl:str) -> pd.DataFrame:
    # Split by |
    T = tl[:typinglog_pipe(tl)]
    # Skip "header" inforamtion
    T = T.split(',', 3)[-1]

    ms = []
    chars = []

    inSpecialChar = False
    lookingForMs = False
    for c in T:
        # NOTE: ASSUMING NO TEXT HAS THE \ CHARACTER
        if not inSpecialChar:
            inSpecialChar = c == '\\'
            if inSpecialChar:
                continue
        else:
            inSpecialChar = c == 'b'
            if not inSpecialChar:
                chars.append(c)
                lookingForMs = True
            continue

        
        if lookingForMs:
            # There is a special case
            #   lookingForMs=True but first digit is '-' (for negative)
            # This is handled by just assuming the next char after an accepted char is part of the ms
            ms.append(c)
            lookingForMs = False
        else:
            if c.isdigit():
                ms[-1] += c
            else:
                chars.append(c)
                lookingForMs = True
        
    ms = [int(m) for m in ms]
    TL = pd.DataFrame({'Char':chars, 'Ms':ms})

    return TL


# The typingLog contains two values for each word
#   its starting index in the full text
#   the number of keystrokes the user took to type the word
# This function is not used to perform parsing (above), but can be used for validation
def parse_typinglog_wordvals(tl:str):
    # Split by |
    T = tl[typinglog_pipe(tl)+1:]
    # Do this to make regex better
    T = ',' + T[:-1]

    # Split into words
    wordnums_pattern = r',(\d+),(\d+),'
    wordnums = re.findall(wordnums_pattern, T)

    text_inds, num_strokes = zip(*wordnums)

    text_inds = [int(i) for i in text_inds]
    num_strokes = [int(i) for i in num_strokes]

    return text_inds, num_strokes
    

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
# Loop through all ways to compute WPM (purportedly)
# Return the wpm with minimum error and its options
def compute_wpm_best(df, wpm_tp):
    opts = ['all', 'no_spaces', 'no_endchar', 'no_endspace']

    min_diff = np.inf
    wpm = np.inf
    opt_t, opt_m = 'all', 'all'
    for m in opts:
        for t in opts:
            wpm_ = compute_wpm_acc(df, t,m)[0]
            wpm_diff = abs(wpm_tp - wpm_)
            if wpm_diff < min_diff:
                min_diff = wpm_diff
                wpm = wpm_
                opt_t, opt_m = t, m

    return wpm, opt_t, opt_m
def compute_acc_best(df, acc_tp):
    opts = ['all', 'no_spaces', 'no_endchar', 'no_endspace']

    min_diff = np.inf
    acc = np.inf
    opt_t, opt_m = 'all', 'all'
    for m in opts:
        for t in opts:
            acc_ = compute_wpm_acc(df, t,m)[1]
            acc_diff = abs(acc_tp - acc_)
            if acc_diff < min_diff:
                min_diff = acc_diff
                acc = acc_
                opt_t, opt_m = t, m

    return acc, opt_t, opt_m
