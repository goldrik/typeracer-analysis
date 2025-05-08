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
    wordprev_ = []
    word_ind = 0
    char_ind = 0
    char_ms, mistake, first_typed_char = reset_word_vars()
    for i in range(num_entries):
        c = TL.iloc[i]['Char']
        char_ind_ = TL.iloc[i]['CharInd']
        op = TL.iloc[i]['Op']

        # print(words[word_ind])
        # print(word_)
        # print(char_ind_)

        if op == '+':
            # word_.append(c)
            word_.insert(char_ind_, c)
            assert(word_[char_ind_] == c)

        elif op == '-':
            assert(word_[char_ind_] == c)
            word_.pop(char_ind_)

        elif op == '$':
            word_[char_ind_] = c
            assert(word_[char_ind_] == c)

        # Accumulate the time needed to type this character
        char_ms += TL.iloc[i]['Ms']

        # if i > 100:
        #     print(word_)
        #     print(words[word_ind][:len(word_)])
        # if i > 190:
            # raise Exception

        # For each character in the word, use the word's substring (so far)
        #   check if the user has typed the substring correctly yet
        # substring = words[word_ind][:char_ind+1]
        substring = words[word_ind][:len(word_)]
        # MUST check for end of keystroke
        #   sometimes the substring my inadvertently match during a keystroke (of deletions)
        keystrokeEnd = (i == num_entries-1) or (TL.index[i] != TL.index[i+1])
        notDelete = op != '-'
        # if (''.join(word_) == substring) and (keystrokeEnd or notDelete):
        if len(word_) > len(wordprev_) and (''.join(word_) == substring) and (keystrokeEnd or TL.iloc[i+1]['Op'] != '-'):
        # if (''.join(word_) == substring) and (len(word_) == char_ind+1) and (keystrokeEnd or notDelete):
        # if ''.join(word_) == substring and keystrokeEnd:
        # if ''.join(word_) == substring and len(word_) == char_ind+1:
            # print(TL.iloc[i])
            # d = lambda w: f"{'-' + w + '-':<12}"
            # print(d(c), d(''.join(word_)), '\t', d(substring), '\t', char_ind, '\t', op)
            # if len(word_) != char_ind+1:
                # raise Exception('checkpoint')
            for wp_ in range(len(wordprev_)):
                assert(word_[wp_] == wordprev_[wp_])
            c_ = ''.join(word_[len(wordprev_):])
            wordprev_ = word_.copy()
            # The correct character
            # c_ = words[word_ind][char_ind]
            if (len(c_) != 1):
                # raise Exception('char to add is not a single character')
                print(f'\tchar to add {c_} is not a single character')
            elif (c_ != words[word_ind][char_ind]):
                raise Exception('char to add is different from usual')
            C_ = len(c_)
            for i_ in range(C_):
                chars_text.append(c_[i_])
                word_inds.append(word_ind)
                
                if i_ == 0:
                    chars_ms.append(char_ms)
                else:
                    chars_ms.append(0)
                mistakes.append(mistake)
                if not mistake: 
                    # Normally, the correct character was typed
                    chars_typed.append(c_)
                else:
                    chars_typed.append(first_typed_char)

                char_ind += 1
            char_ms, mistake, first_typed_char = reset_word_vars()

            # If we've reached the end of the text or end of word
            if len(word_):
                if ''.join(word_) == words[word_ind]:
                # if (i == num_entries-1) or \
                    # ((c == ' ') and (TL.iloc[i+1]['CharInd'] == 0) and (TL.iloc[i+1]['Op'] == '+')):
                    text_ += ''.join(word_)
                    word_ = []
                    wordprev_ = []

                    word_ind += 1
                    char_ind = 0
                    char_ms, mistake, first_typed_char = reset_word_vars()
        else:
            mistake = True
            # Set this variable for just the first character
            if first_typed_char == '':
                first_typed_char = c

        # # If we've reached the end of the text or end of word
        # if len(word_):
        #     if ''.join(word_) == words[word_ind]:
        #     # if (i == num_entries-1) or \
        #         # ((c == ' ') and (TL.iloc[i+1]['CharInd'] == 0) and (TL.iloc[i+1]['Op'] == '+')):
        #         text_ += ''.join(word_)
        #         word_ = []
        #         wordprev_ = []

        #         word_ind += 1
        #         char_ind = 0
        #         char_ms, mistake, first_typed_char = reset_word_vars()

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
    T = tl[typinglog_pipe(tl)+1:]
    # Do this to make regex better
    T = ',' + T[:-1]


    ## WINDOWS

    # Split into windows
    wordnums_pattern = r',(\d+),(\d+),'
    # This makes it so regex matches are all unique (no overlaps)
    wordnums_pattern_ = wordnums_pattern.replace('(', '').replace(')', '')

    # Each window starts with \d,\d
    wordnums = re.findall(wordnums_pattern, T)
    # These are the indices in the original text and the number of strokes per window
    text_inds, num_strokes = zip(*wordnums)
    text_inds = [int(i) for i in text_inds]
    num_strokes = [int(i) for i in num_strokes]

    # Get the keystrokes per window
    windows = re.split(wordnums_pattern_, T)[1:]
    # * Implicity checks for equality of length (windows processed correctly)
    W = pd.DataFrame({'Window': windows, 'NumStrokes': num_strokes, 'TextInd': text_inds})


    ## KEYSTROKES

    # Each window now comprises of multiple keystrokes
    # Each keystroke is comprised of one or more entries
    # Each entry has this pattern
    #   ,{ms},{kestroke},
    # Use this pattern to separate out each keystroke
    #   Looking for the ms is more robust than finding keystrokes
    ms_pattern = r',-?\d+,'

    # For each keystroke
    strokes, strokes_ms, window_nums = ([] for _ in range(3))

    for w in W.itertuples():
        # For regex robustness
        window_ = ',' + w.Window
        ms_vals = re.findall(ms_pattern, window_)
        keystrokes = re.split(ms_pattern, window_)[1:]

        # Window contains multiple keystrokes
        for s, (ms_, stroke) in enumerate(zip(ms_vals, keystrokes)):
            ms = int(ms_[1:-1])

            strokes.append(stroke)
            strokes_ms.append(ms)
            window_nums.append(w.Index)

    S = pd.DataFrame({'Stroke': strokes, 'Ms': strokes_ms, 'Window': window_nums})


    ## ENTRIES

    # Entry(ies) can be found in each keystroke
    # Each keystroke has this pattern
    #   ,({ind}{op}{char})+,
    # * "ind" is the "window_ind", or the index in the window being typed
    char_pattern = r'(\d+)([+\-$])(\\"|.)'

    window_inds, chars, ops, char_ms, window_nums, stroke_nums, stroke_inds = \
        ([] for _ in range(7))

    for s in S.itertuples():
        entries = re.findall(char_pattern, s.Stroke)
        num_entries = len(entries)
        assert(num_entries > 0)

        # NOTE: Two ways of saving ms for keystrokes w multiple entries
        # 1. First entry gets the whole ms, other entries get 0
        ms = s.Ms
        # # 2. Divide between number of entries
        # ms = s.Ms / num_entries

        # This is which stroke (from the strokes DataFrame) the entry(ies) come from
        stroke_num = s.Index
        # This is the window that the stroke came from
        window_num = s.Window

        for i, (char_ind_, op, char) in enumerate(entries):
            char_ind = int(char_ind_)

            # The "character" found with regex can be \" (two characters), so take the last index
            chars.append(char[-1])
            ops.append(op)
            # This is "purportly" the index within the full window
            # ! This is from typingLog, so can be buggy!
            window_inds.append(char_ind)
            
            char_ms.append(ms)

            stroke_nums.append(stroke_num)
            # Individual entry's index in the stroke
            stroke_inds.append(i)

            window_nums.append(window_num)
            
            # Only save the ms once (to avoid double-dipping)
            # Only relevant to first way of saving Ms (see above)
            ms = 0

    TL = pd.DataFrame({'Window': window_nums, 'WindowInd': window_inds, 'Char': chars, 'Op': ops, 'Ms': char_ms, 'Stroke': stroke_nums, 'StrokeInd': stroke_inds})
    return TL, S, W


def reconstruct_text_typinglog(TL):
    t = ''
    # T = len(TL)

    for _,W in TL.groupby('Window'):
        window = []
        word_ind_offset = 0
        for i,w in enumerate(W.itertuples()):
            c = w.Char
            word_ind = w.WindowInd
            op = w.Op

            if i == 0 and len(t) and t[-1] != ' ':
                # Find the index of the last space in t
                spaceInd = t.rfind(' ')
                t_ = t[spaceInd+1:]
                t = t[:spaceInd+1]
                window = window + list(t_)
            
            word_ind = word_ind + word_ind_offset

            if len(window) < word_ind:
                window += [''] * (word_ind - len(window))

            if op == '+':
                window.insert(word_ind, c)
                # assert(window[word_ind] == c)

            elif op == '-':
                # assert(window[word_ind] == c)
                window.pop(word_ind)

            elif op == '$':
                window[word_ind] = c
                # assert(window[word_ind] == c)
        

        # if len(t) and t[-1] != ' ':
        #     for w in range(1, len(window)+1):
        #         window_ = ''.join(window[:w])
        #         if len(t) > w:
        #             t_ = t[-w:]
        #         else:
        #             break
        #         if window_ == t_:
        #             t = t[:-w]
        #             break
        t += ''.join(window)

    # Check if there was a duplicate punctuation at the end of the text (TypeRacer bug, or input lag perhaps)
    # The duplicate characters should be from the same keystroke (probably)
    if t[-1] == t[-2]:
        if TL.iloc[-1]['Stroke'] == TL.iloc[-2]['Stroke']:
            # Ignore the ellipses case
            if t[-3:] != '...':
                t = t[:-1]

    return t


def reconstruct_text_typinglog__(TL):
    t = ''
    # T = len(TL)

    for _,W in TL.groupby('Window'):
        window = []
        for i,w in enumerate(W.itertuples()):
            c = w.Char
            word_ind = w.WindowInd
            op = w.Op

            if i == 0 and op != '+':
                # Probably trying to delete a character from the previous word
                # Make sure the characters match of course
                if t[-1] == c and op == '-':
                    t = t[:-1]
                continue

            if len(window) < word_ind:
                window += [''] * (word_ind - len(window))

            if op == '+':
                window.insert(word_ind, c)
                # assert(window[word_ind] == c)

            elif op == '-':
                # assert(window[word_ind] == c)
                window.pop(word_ind)

            elif op == '$':
                window[word_ind] = c
                # assert(window[word_ind] == c)
        

        if len(t) and t[-1] != ' ':
            for w in range(1, len(window)+1):
                window_ = ''.join(window[:w])
                if len(t) > w:
                    t_ = t[-w:]
                else:
                    break
                if window_ == t_:
                    t = t[:-w]
                    break
        t += ''.join(window)

    return t


def reconstruct_text_typinglog_(TL):
    T = len(TL)
    t = ''
    word = []
    for i in range(T):
        c = TL.iloc[i]['Char']
        word_ind = TL.iloc[i]['CharInd']
        op = TL.iloc[i]['Op']

        if op == '+':
            word.append(c)
            assert(word[word_ind] == c)

        elif op == '-':
            assert(word[word_ind] == c)
            word.pop(word_ind)

        elif op == '$':
            word[word_ind] = c
            assert(word[word_ind] == c)

        if (i == T-1) or ((c == ' ') and (TL.iloc[i+1]['CharInd'] == 0) and (TL.iloc[i+1]['Op'] == '+')):
            t += ''.join(word)
            word = []

    return t



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
