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
# TYPINGLOG

# The typingLog contains two formats, separated by a pipe character
# To account for the possiblity of a pipe in the text itself, search for a specific regex pattern
#   i.e. pipe character followed by digits (first word index, should always be 0)
def clean_typinglog(tl: str) -> str:
    tl = tl.replace('\\\\', '\\')
    tl = tl.replace('\\"', '"')

    unicode_pattern = '\\\\u[0-9a-fA-F]{4}'
    unicode_occurences = re.finditer(unicode_pattern, tl)
    for unicode_occurence in unicode_occurences:
        unicode_str = unicode_occurence.group()
        tl = tl.replace(unicode_str, unicode_str.encode('utf-8').decode('unicode-escape'))

    return tl

def reverse_char_clean(c: str) -> str:
    assert(len(c) == 1)
    if c == '\\':
        return '\\\\'
    elif c == '"':
        return '\\"'
    else:
        return c.encode('unicode-escape').decode('utf-8').replace('\\x', '\\u00')

# The typingLog contains two formats, separated by a pipe character
# To account for the possiblity of a pipe in the text itself, search for a specific regex pattern
#   i.e. pipe character followed by digits (first word index, should always be 0)
def typinglog_pipe(tl: str) -> int:
    pipe_pattern = r'\|\d+,'
    pipe_occurence = list(re.finditer(pipe_pattern, tl))

    return pipe_occurence[-1].start()


# Outputs three dataframes: all keystrokes, text characters, words
def parse_typinglog(tl):
    # This gives us all the keystrokes/entries first
    TL = parse_typinglog_complete(tl)[0]

    TL_ = TL.copy()
    TL_ = split_typinglog_addition_entries(TL_)

    text, partials = reconstruct_text_typinglog(TL_)

    # TypingLog DataFrame arrays
    # Save as numpy arrays instead for MUCH faster iteration
    tl_window_inds = TL_['WindowInd'].to_numpy()
    tl_chars = TL_['Char'].to_numpy()
    tl_ops = TL_['Op'].to_numpy()
    tl_ms = TL_['Ms'].to_numpy()
    tl_strokes = TL_['Stroke'].to_numpy()


    ## CHARACTERS

    # For each character in the text
    chars_text = []
    # Time it took to type each character CORRECTLY
    # i.e. includes time for incorrect keystrokes
    chars_ms = []
    # Was there a mistake made when typing this character
    mistakes = []
    # Which character did the user type
    chars_typed = []

    # When a character is typed correctly, reset the variables
    def reset_vars(): return 0, False, ''

    window_ind_max = np.max(tl_window_inds)
    # Keep track of how much text has been typed correctly
    text_ind = 0
    # Keep track of keystrokes needed for each correct character typed
    keystrokes = []
    # Loop through each keystroke
    for s,partial in enumerate(partials):
        lenPartial = len(partial)
        # *After* each stroke, check if the next portion of the text has been typed
        #   NOTE: Make sure the new text is longer than previous 
        #         (keystrokes may include deletions of correct characters)

        # Track the keystrokes required
        keystrokes.append(s)
        if lenPartial > text_ind and partial == text[:lenPartial]:
            # New *correct* characters that have been typed
            new_chars = text[text_ind:lenPartial]
            text_ind = lenPartial

            # Keystrokes required -> all typed entries (additions, deletions, etc)
            keystrokes = np.array(keystrokes)
            inds_strokes = np.isin(tl_strokes, keystrokes)
            num_entries = np.count_nonzero(inds_strokes)
            # S = TL_[inds_strokes]


            # Given all the keystrokes needed to type the new characters
            # Iterate through the individual entries to see when each character got typed correctly
            #   This handles cases where multiple new characters were typed at once

            # Simulate TypeRacer window for just these keystrokes
            #   Extra indices provided since keystrokes dont necessarily start from index 0
            window = [''] * window_ind_max

            # Keep track of when the previous character was typed correctly
            entry_ind = -1
            new_chars_ = new_chars
            char_ms, mistake, char_typed = reset_vars()
            nc = 1
            for i,(ind, c, op, ms) in enumerate(zip( \
                    tl_window_inds[inds_strokes], tl_chars[inds_strokes], tl_ops[inds_strokes], tl_ms[inds_strokes]) ):
                # BY DEFINITION, the last new character was typed at the end of the keystrokes
                #   This is given by the way we defined this loop (partials)
                # if len(new_chars_) == 1:
                if nc == len(new_chars):
                    break
                # If last entry, let the for loop (below) handle it
                if i == num_entries-1:
                    break

                # Track the amount of time
                char_ms += ms
                # Character typed
                if char_typed == '':
                    char_typed = c

                # Update window with new entry
                if op == '+':
                    window.insert(ind, c)
                elif op == '-':
                    window.pop(ind)
                elif op == '$':
                    window[ind] = c

                window_text = ''.join(window)
                # Check if the latest typed character matches
                # if len(window_text) == (len(new_chars)-len(new_chars_)+1) and window_text[-1] == new_chars_[0]:
                if window_text == new_chars[:nc]:
                    # Do not count the entry if it's a deletion (unless at end of keystroke)
                    # if op != '-' or S['Stroke'].iloc[i+1] != S['Stroke'].iloc[i]:
                    if op != '-' or tl_strokes[inds_strokes][i+1] != tl_strokes[inds_strokes][i]:
                        entry_ind = i
                        
                        # chars_text.append(new_chars_[0])
                        # new_chars_ = new_chars_[1:]
                        chars_text.append(new_chars[nc-1])
                        nc += 1

                        chars_ms.append(char_ms)
                        mistakes.append(mistake)
                        chars_typed.append(char_typed)

                        char_ms, mistake, char_typed = reset_vars()

                else:
                    # If the first entry did not match, then mark a mistake occurred
                    mistake = True

            # The rest of the chars
            new_chars_ = new_chars[nc-1:]
            nc = len(new_chars_)
            # If the number of new characters matches the number of entries, no mistake
            mistake = nc != np.count_nonzero(inds_strokes)-(entry_ind+1)
            ms = np.sum(tl_ms[inds_strokes][entry_ind+1:])
            for c in new_chars_:
                chars_text.append(c)
                chars_ms.append(ms)
                # chars_ms.append(ms / nc)
                mistakes.append(mistake)
                if mistake:
                    chars_typed.append(tl_chars[inds_strokes][entry_ind+1])
                else:
                    chars_typed.append(c)

                ms = 0


            # RESET for next character
            keystrokes = []


    # Which word does each character belong to, as well as index within word
    word_nums = [0]
    word_inds = [0]
    # Look at previous character
    for c in chars_text[:-1]:
        if c == ' ':
            word_nums.append(word_nums[-1]+1)
            word_inds.append(0)
        else:
            word_nums.append(word_nums[-1])
            word_inds.append(word_inds[-1]+1)
    
    C = pd.DataFrame({'Char':chars_text, 'Word':word_nums, 'WordInd':word_inds, 'Ms':chars_ms, 'Mistake':mistakes, 'Typed':chars_typed})


    ## WORDS

    words=  text.split(' ')
    words[:-1] = [w+' ' for w in words[:-1]]

    words_ms = []
    # The full attempt for this word
    word_mistakes = []
    word_lens = []

    for _,W in C.groupby('Word'):
        words_ms.append(W['Ms'].sum())
        word_mistakes.append(W['Mistake'].sum())
        word_lens.append(len(W))

    # This will automatically check that 
    #   the number of words from the text (words list) matches the number of words found in C (other lists)
    # W = pd.DataFrame({'Attempt':word_keystrokes, 'Keystrokes':word_strokes})
    W = pd.DataFrame({'Word':words, 'Ms':words_ms, 'Mistakes':word_mistakes, 'Length':word_lens})

    return TL, C, W, text


# Parse the second half of typingLog
#   This contains *all* the keystrokes
def parse_typinglog_complete(tl:str):
    T = tl[typinglog_pipe(tl)+1:]
    # Do this to make regex better
    T = ',' + T[:-1]
    T = clean_typinglog(T)


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
    char_pattern = r'(\d+)([+\-$])(.)'

    # The string contains substrings of the form \u00e5
    #   where the u is literal, and the following four characters are hexadecimal digits
    #   this is a Unicode escape sequence
    #   The regex for this is \\u[0-9a-fA-F]{4}

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
    char_pattern = r'(\d+)([+\-$])(.)'

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
            chars.append(char)
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


# Parse the second half of typingLog
#   This contains *all* the keystrokes
def parse_typinglog_complete_(tl:str):
    T = tl[typinglog_pipe(tl)+1:]
    # Do this to make regex better
    T = ',' + T[:-1]
    T = clean_typinglog(T)


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
    # W = pd.DataFrame({'Window': windows, 'NumStrokes': num_strokes, 'TextInd': text_inds})


    ## KEYSTROKES

    # Each window now comprises of multiple keystrokes
    # Each keystroke is comprised of one or more entries
    # Each entry has this pattern
    #   ,{ms},{kestroke},
    # Use this pattern to separate out each keystroke
    #   Looking for the ms is more robust than finding keystrokes
    ms_pattern = r',-?\d+,'
    char_pattern = r'(\d+)([+\-$])(.)'

    # For each keystroke
    # strokes, strokes_ms, window_nums = ([] for _ in range(3))

    window_inds, chars, ops, char_ms, window_nums, stroke_nums, stroke_inds = \
        ([] for _ in range(7))

    ss = -1
    ee = -1

    # for w in W.itertuples():
    for w in range(len(windows)):
        # For regex robustness
        window_ = ',' + windows[w]
        ms_vals = re.findall(ms_pattern, window_)
        keystrokes = re.split(ms_pattern, window_)[1:]

        # Window contains multiple keystrokes
        for s, (ms_, stroke) in enumerate(zip(ms_vals, keystrokes)):
            ss += 1
            ms = int(ms_[1:-1])

            entries = re.findall(char_pattern, stroke)
            num_entries = len(entries)
            assert(num_entries > 0)

            # NOTE: Two ways of saving ms for keystrokes w multiple entries
            # 1. First entry gets the whole ms, other entries get 0
            ms = int(ms_[1:-1])

            # This is which stroke (from the strokes DataFrame) the entry(ies) come from
            stroke_num = ss
            # This is the window that the stroke came from
            window_num = w

            # # 2. Divide between number of entries
            # ms = s.Ms / num_entries

            for i, (char_ind_, op, char) in enumerate(entries):
                ee += 1
                char_ind = int(char_ind_)

                # The "character" found with regex can be \" (two characters), so take the last index
                chars.append(char)
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

            # strokes.append(stroke)
            # strokes_ms.append(ms)
            # window_nums.append(w)

    # S = pd.DataFrame({'Stroke': strokes, 'Ms': strokes_ms, 'Window': window_nums})


    ## ENTRIES

    # Entry(ies) can be found in each keystroke
    # Each keystroke has this pattern
    #   ,({ind}{op}{char})+,
    # * "ind" is the "window_ind", or the index in the window being typed


    # for s in S.itertuples():




    TL = pd.DataFrame({'Window': window_nums, 'WindowInd': window_inds, 'Char': chars, 'Op': ops, 'Ms': char_ms, 'Stroke': stroke_nums, 'StrokeInd': stroke_inds})
    # return TL, S, W
    return TL, 0, 0


# The typingLog contains some keystrokes consisting of multiple entries
# Some such keystrokes are just multiple additions in a row
# To make typingLog parsing eeasier, split these into multiple keystrokes
def split_typinglog_addition_entries(TL):
    # For speed, use numpy arrays
    strokes = TL['Stroke'].to_numpy()
    stroke_inds = TL['StrokeInd'].to_numpy()
    chars = TL['Char'].to_numpy()
    ops = TL['Op'].to_numpy()
    mss = TL['Ms'].to_numpy()

    # When we split keystrokes, the keystroke numbers will increase for the remaining keystrokes
    # Keep track of that increase
    stroke_offset = 0
    # Compute indices upfront, since we will be editing the strokes array
    stroke_inds_all = [strokes == s for s in range(strokes[-1]+1)]
    for s,inds in enumerate(stroke_inds_all):
        strokes[inds] += stroke_offset
        # Number of entries
        ne = np.count_nonzero(inds)
        # Normal keystroke (w one entry), ignore
        if ne == 1:
            continue

        # Special case: duplicate punctuation at the end of the text stored as single keystroke
        if s == len(stroke_inds_all)-1:
            stroke_full = ''.join(chars[inds])
            if not stroke_full.endswith('...'): 
                if chars[inds][-1] == chars[inds][-2]:
                    continue

        if np.all(ops[inds] == '+'):
            # Split keystroke
            strokes[inds] = strokes[inds] + np.arange(ne)
            stroke_offset += (ne-1)

            stroke_inds[inds] = 0
            # TODO: Should handle milliseconds as well (divide evenly? instead of just setting to 0)

    TL['Stroke'] = strokes
    TL['StrokeInd'] = stroke_inds
    TL['Op'] = ops
    TL['Ms'] = mss
    return TL


# Reconstruct text from typingLog DataFrame (containing keystrokes)
def reconstruct_text_typinglog(TL):
    # Keep track of text entered thus far (including incorrect chars)
    entries = []
    # After each keystroke, record the state of the text
    states = []

    windows = TL['Window'].to_numpy()

    tl_window_inds = TL['WindowInd'].to_numpy()
    tl_chars = TL['Char'].to_numpy()
    tl_ops = TL['Op'].to_numpy()
    tl_strokes = TL['Stroke'].to_numpy()

    # for _,W in TL.groupby('Window'):
    for w in range(windows[-1]+1):
        tl_inds = windows == w

        # Add a dummy stroke at the end (to simulate next new keystroke)
        # strokes = W['Stroke'].to_list() + [-1]
        strokes = list(tl_strokes[tl_inds]) + [-1]

        window_start_ind = len(entries)
        if len(entries) and entries[-1] != ' ':
            # Move the window back to the last space
            window_start_ind = len(entries) - entries[::-1].index(' ')
        # for i,w in enumerate(W.itertuples()):
        for i,(ind, c, op) in enumerate(zip(tl_window_inds[tl_inds], tl_chars[tl_inds], tl_ops[tl_inds])): 
            text_ind = ind + window_start_ind

            if op == '+':
                entries.insert(text_ind, c)
                assert(entries[text_ind] == c)

            elif op == '-':
                assert(entries[text_ind] == c)
                entries.pop(text_ind)

            elif op == '$':
                entries[text_ind] = c
                assert(entries[text_ind] == c)

            # If we're at the end of a stroke, THEN record state of text
            if strokes[i+1] != strokes[i]:
                states.append(''.join(entries))
    

    text = ''.join(entries)
    # Check if there was a duplicate punctuation at the end of the text (TypeRacer bug, or input lag perhaps)
    # The duplicate characters should be from the same keystroke (probably)
    # Ignore the ellipses case
    if len(text) > 3:
        if text[-1] == text[-2] and TL.iloc[-1]['Stroke'] == TL.iloc[-2]['Stroke'] and text[-3:] != '...':
            text = text[:-1]
            states[-1] = states[-1][:-1]

    return text, states


# Parse the first half of typingLog
# This contains all the correct characters and the time it took to *correctly* type each
#   * not all entries, only correct ones shown
def parse_typinglog_simple(tl:str) -> pd.DataFrame:
    # Split by |
    T = tl[:typinglog_pipe(tl)]
    # Skip "header" inforamtion
    T = T.split(',', 3)[-1]
    T =  clean_typinglog(T)

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


