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
def parse_typinglog_(tl):
    # This gives us all the keystrokes/entries first
    TL = parse_typinglog_complete(tl)[0]
    text, textPartials = reconstruct_text_typinglog(TL)
    num_strokes = TL['Stroke'].iloc[-1] + 1
    num_entries = len(TL)


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

    def text_match_inds(t, tp):
        text_ind = 0
        inds = np.full(len(tp), False)
        for i,tp_ in enumerate(tp):
            nTp = len(tp_)
            if nTp > text_ind:
                if tp_ == t[:nTp]:
                    text_ind = nTp
                    inds[i] = True
        return inds

    inds = text_match_inds(text, textPartials)

    def reset_vars(): return 0, False, ''
    ms, mistake, char_typed = reset_vars()
    text_ind = 0
    strokes = []
    for e in range(num_entries):
        TL_ = TL.iloc[e]
        # For each keystroke, check if we've matched the text thus far
        # The text thus far has to be longer than text before
        # If we've got a match, see how many keystrokes it took to get there
        # For those keystrokes, 
        #   if only one entry -> no mistake
        #   first added char is typed char
        #   add all ms
        #     divide by number of text chars?

        ms += TL_['Ms']
        if char_typed == '':
            char_typed = TL_['Char']

        tp = textPartials[e]
        nTp = len(tp)
        print(e)
        print(tp)
        print(text[:nTp])
        if nTp > text_ind and tp == text[:nTp]:
            print('\tTrue')
            newChar = text[text_ind]
            assert(len(newChar) == 1)
            text_ind = nTp

            chars_text.append(newChar)
            chars_ms.append(ms)
            mistakes.append(mistake)
            chars_typed.append(char_typed)
        else:
            mistake = True


    word_inds = [0]
    for c in chars_text[1:]:
        if c == ' ':
            word_inds.append(word_inds[-1]+1)
        else:
            word_inds.append(word_inds[-1])
    
    C = pd.DataFrame({'WordInd':word_inds, 'Char':chars_text, 'Ms':chars_ms, 'Mistake':mistakes, 'Typed':chars_typed})

    ## WORDS

    words=  text.split(' ')
    words[:-1] = [w+' ' for w in words[:-1]]

    words_ms = []
    # The full attempt for this word
    word_mistakes = []
    word_lens = []

    for _,W in C.groupby('WordInd'):
        words_ms.append(W['Ms'].sum())
        word_mistakes.append(W['Mistake'].sum())
        word_lens.append(len(W))

    # W = pd.DataFrame({'Word':words, 'Attempt':word_keystrokes, 'Ms':words_ms, 'Mistake':word_mistakes, 'Length':word_lens, 'Keystrokes':word_strokes})
    W = pd.DataFrame({'Word':words, 'Ms':words_ms, 'Mistake':word_mistakes, 'Length':word_lens})

    return TL, C, W, text


# Outputs three dataframes: all keystrokes, text characters, words
def parse_typinglog(tl):
    # This gives us all the keystrokes/entries first
    TL = parse_typinglog_complete(tl)[0]
    text, textPartials = reconstruct_text_typinglog(TL)


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

    def reset_vars(): return 0, False, ''

    tl_strokes = TL['Stroke'].to_numpy()
    window_ind_max = TL['WindowInd'].max()


    # Keep track of how much text has been typed correctly
    text_ind = 0
    # Keep track of keystrokes needed for each correct character typed
    keystrokes = []
    # Loop through each keystroke
    for s,tp in enumerate(textPartials):
        lenPartial = len(tp)
        # *After* each stroke, check if the next portion of the text has been typed
        #   NOTE: Make sure the new text is longer than previous 
        #         (keystrokes may include deletions of correct characters)

        # For those keystrokes, 
        #   if only one entry -> no mistake
        #   first added char is typed char
        #   add all ms
        #     divide by number of text chars?

        # Track the keystrokes required
        keystrokes.append(s)
        if lenPartial > text_ind and tp == text[:lenPartial]:
            # New *correct* characters that have been typed
            new_chars = text[text_ind:lenPartial]
            text_ind = lenPartial

            # Keystrokes required -> all typed entries (additions, deletions, etc)
            keystrokes = np.array(keystrokes)
            TL_ = TL[np.isin(tl_strokes, keystrokes)]
            # |
            # v
            char_ms, mistake, char_typed = reset_vars()

            inds = TL_['WindowInd'].to_list()
            chars = TL_['Char'].to_list()
            ops = TL_['Op'].to_list()
            entry_ms = TL_['Ms'].to_list()

            # Given all the keystrokes needed to type the new characters
            # Iterate through the individual entries to see when each character got typed correctly
            #   This handles cases where multiple new characters were typed at once

            # Simulate TypeRacer window for just these keystrokes
            #   Extra indices provided since keystrokes dont necessarily start from index 0
            window = [''] * window_ind_max

            # Keep track of when the previous character was typed correctly
            entry_ind = -1
            new_chars_ = new_chars
            for i,(ind, c, op, ms) in enumerate(zip(inds, chars, ops, entry_ms)):
                # BY DEFINITION, the last new character was typed at the end of the keystrokes
                #   This is given by the way we defined this loop (textPartials)
                if len(new_chars_) == 1:
                    break
                if i == len(TL_)-1:
                    break

                # Track the amount of time
                char_ms += ms
                # Character typed
                if char_typed == '':
                    char_typed = c
                    if op == '-':
                        print('something')

                # Update window with new entry
                if op == '+':
                    window.insert(ind, c)
                elif op == '-':
                    window.pop(ind)
                elif op == '$':
                    window[ind] = c

                window_text = ''.join(window)
                # Check if the latest typed character matches
                if len(window_text) and window_text[-1] == new_chars_[0]:
                    # Do not count the entry if it's a deletion (unless at end of keystroke)
                    if op != '-' or TL_['Stroke'].iloc[i+1] != TL_['Stroke'].iloc[i]:
                        entry_ind = i

                        chars_text.append(new_chars_[0])
                        new_chars_ = new_chars_[1:]

                        chars_ms.append(char_ms)
                        mistakes.append(mistake)
                        chars_typed.append(char_typed)

                        char_ms, mistake, char_typed = reset_vars()

                # If the first entry did not match, then mark a mistake occurred
                else:
                    mistake = True

            # The rest of the chars
            nc = len(new_chars_)
            # If the number of new characters matches the number of entries, no mistake
            mistake = nc != len(TL_)-(entry_ind+1)
            for c in new_chars_:
                chars_text.append(c)
                # chars_ms.append(entry_ms.iloc[entry_ind+1:].sum() / nc)
                chars_ms.append(np.sum(entry_ms[entry_ind+1:]) / nc)
                mistakes.append(mistake)
                if mistake:
                    # chars_typed.append(chars.iloc[entry_ind+1])
                    chars_typed.append(chars[entry_ind+1])
                else:
                    chars_typed.append(c)

            keystrokes = []

    word_inds = [0]
    for c in chars_text[1:]:
        if c == ' ':
            word_inds.append(word_inds[-1]+1)
        else:
            word_inds.append(word_inds[-1])
    
    C = pd.DataFrame({'WordInd':word_inds, 'Char':chars_text, 'Ms':chars_ms, 'Mistake':mistakes, 'Typed':chars_typed})


    ## WORDS

    words=  text.split(' ')
    words[:-1] = [w+' ' for w in words[:-1]]

    words_ms = []
    # The full attempt for this word
    word_mistakes = []
    word_lens = []

    for _,W in C.groupby('WordInd'):
        words_ms.append(W['Ms'].sum())
        word_mistakes.append(W['Mistake'].sum())
        word_lens.append(len(W))

    # W = pd.DataFrame({'Word':words, 'Attempt':word_keystrokes, 'Ms':words_ms, 'Mistake':word_mistakes, 'Length':word_lens, 'Keystrokes':word_strokes})
    W = pd.DataFrame({'Word':words, 'Ms':words_ms, 'Mistake':word_mistakes, 'Length':word_lens})

    return TL, C, W, text


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
        # for i,w in enumerate(W.itertuples()):
        for i,(ind, c, op) in enumerate(zip(tl_window_inds[tl_inds], tl_chars[tl_inds], tl_ops[tl_inds])): 
            if i == 0 and len(entries) and entries[-1] != ' ':
                # Move the window back to the last space
                window_start_ind = len(entries) - entries[::-1].index(' ')
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


