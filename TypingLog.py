# Aurik Sarker
# 09 May 2025

import pandas as pd

from typeracer_utils import *
from parse_soup import *


#%%


# Stores the typingLog string and facilitates a variety of processes for it
# The processes are used to generate DataFrames

# User input: User inputs into the TypeRacer input field, the "window"
#             Each input is an "entry", an addition/deletion/substitution of a single character
#             The entries are sometimes grouped into a "keystroke"
#               when multiple characters are selected and deleted/substituted at once
#               when multiple entries occur before being recorded by TypeRacer (due to lag)
#             Keystrokes fill up the "window", which is what the user sees in the input field
#               Each window SHOULD end with the entry of space after the completion of a word
#                 BUT this is not always true due to input lag

# Output: The user inputs all add up in the end to the target text (typedText)
#         Each character is marked as complete once it is typed correctly
#         A word is complete when 

# Input user keystrokes
#   entries - 
#   strokes - 
#   windows - 

# Output text characters
#   _chars - 
#   chars - 
#   words - 

_PATTERN_WORDNUMS = r',(\d+),(\d+),'
_PATTERN_MS = r',-?\d+,'
_PATTERN_ENTRIES = r'(\d+)([+\-$])(.)'


class TypingLog:
    """TypingLog"""

    def __init__(self, tl: str):
        # Initialize DataFrame vars
        self.clear_data()
        self.text = ''

        # Full typinglog string
        self.tl = tl

        # Get header
        # The substring preceding the third comma
        self.header = tl[:[ind for ind, c in enumerate(tl) if c == ','][2]]

        # For internal use
        # Separates the typingLog into its two parts (separated by a pipe character)
        # "Cleans" the string for escaped characters (\\, \", unicode)
        ind = TypingLog.typinglog_pipe(tl)
        self._tl = [ TypingLog.clean_typinglog(_tl) for _tl in 
                           [tl[:ind], tl[ind+1:]] ]
        
    
    # This generates the entries DataFrame and the two "output" DataFrames (chars and words)
    # If that DataFrame is empty, calls parse_entries() to generate it
    #   This function also creates the other two "input" DataFrames (strokes and windows)
    def generate(self):
        if self.entries.empty:
            self.parse_entries()
        if self._chars.empty:
            self.parse_chars()
        
        # To make processing easier, split these problematic strokes (with consecutive additions)
        entries_ = TypingLog.adjust_end_entries(self._chars, self.entries)
        entries_ = TypingLog.split_addition_entries(entries_)
        # Then recreate the text input
        text, partials = TypingLog.generate_typed_input(entries_)
        self.text = text

        assert(text == partials[-1])

        # TypingLog DataFrame arrays
        # Save as numpy arrays instead for MUCH faster iteration though DataFrame
        tl_window_inds = entries_['WindowInd'].to_numpy()
        tl_chars = entries_['Char'].to_numpy()
        tl_ops = entries_['Op'].to_numpy()
        tl_ms = entries_['Ms'].to_numpy()
        tl_strokes = entries_['Stroke'].to_numpy()


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


        # The window needs to have enough indices to handle any entry
        len_window = np.max(tl_window_inds)
        # Keep track of how much text has been typed correctly thus far
        len_text_sofar = 0
        # Use this to record which keystrokes were needed to type the next partial text
        keystrokes = []
        
        # When a character is typed correctly, reset the variables
        def reset_vars(): return 0, False, ''

        # Loop through each keystroke
        for s,partial in enumerate(partials):
            len_partial = len(partial)
            # Record this keystroke
            keystrokes.append(s)

            # *After* each stroke, check if the next portion of the text has been typed
            #   NOTE: Make sure the new text is longer than previous 
            #         (keystrokes may include deletions of correct characters)
            if len_partial > len_text_sofar and partial == text[:len_partial]:
                # New (correct) text characters that have been typed
                new_chars = text[len_text_sofar:len_partial]
                len_text_sofar = len_partial

                # Indices (in entries) for all entries corresponding to these keystrokes
                inds_S = np.isin(tl_strokes, np.array(keystrokes))
                # S = entries_[inds_S]

                # Given all the keystrokes needed to type the new characters
                # Iterate through the individual entries to see when each character got typed correctly
                #   This handles cases where multiple new characters were typed at once

                # Simulate TypeRacer window for just these keystrokes
                #   Extra indices provided since keystrokes dont necessarily start from index 0
                window = [''] * len_window

                # Keep track of when the previous character was typed correctly
                ind_entry = -1
                chars_added = 0
                char_ms, mistake, char_typed = reset_vars()
                for i,(ind, c, op, ms) in enumerate(zip( \
                        tl_window_inds[inds_S], tl_chars[inds_S], tl_ops[inds_S], tl_ms[inds_S]) ):
                    # BY DEFINITION, the last new character was typed at the end of the keystrokes
                    #   This is given by the way we defined this loop (partials)
                    if chars_added == len(new_chars)-1:
                        break
                    # If last entry, let the for loop (below) handle it
                    if i == np.count_nonzero(inds_S)-1:
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
                    else:
                        raise Exception('ERROR: Unexpected operation symbol found in typingLog')

                    window_text = ''.join(window)
                    # Check if the latest typed character matches
                    if window_text == new_chars[:chars_added+1]:
                        # Do not count the entry if it's a deletion (unless at end of keystroke)
                        # if op != '-' or S['Stroke'].iloc[i+1] != S['Stroke'].iloc[i]:
                        if op != '-' or tl_strokes[inds_S][i+1] != tl_strokes[inds_S][i]:
                            ind_entry = i
                            
                            chars_text.append(new_chars[chars_added])
                            chars_added += 1

                            chars_ms.append(char_ms)
                            mistakes.append(mistake)
                            chars_typed.append(char_typed)

                            char_ms, mistake, char_typed = reset_vars()

                    else:
                        # If the first entry did not match, then mark a mistake occurred
                        mistake = True

                # The rest of the chars
                new_chars_ = new_chars[chars_added:]
                nc = len(new_chars_)
                # If the number of new characters matches the number of entries, no mistake
                all_mistakes = nc != np.count_nonzero(inds_S)-(ind_entry+1)
                # Sum the remaining time
                all_ms = np.sum(tl_ms[inds_S][ind_entry+1:])
                for c in new_chars_:
                    chars_text.append(c)

                    mistakes.append(all_mistakes)
                    if all_mistakes:
                        chars_typed.append(tl_chars[inds_S][ind_entry+1])
                    else:
                        chars_typed.append(c)

                    # chars_ms.append(all_ms / nc)
                    chars_ms.append(all_ms)
                    all_ms = 0

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
        
        self.chars = pd.DataFrame({'Char':chars_text, 'Word':word_nums, 'WordInd':word_inds, 'Ms':chars_ms, 'Mistake':mistakes, 'Typed':chars_typed})


        ## WORDS
        words=  text.split(' ')
        words[:-1] = [w+' ' for w in words[:-1]]

        words_ms = []
        # The full attempt for this word
        word_mistakes = []
        word_lens = []

        for _,W in self.chars.groupby('Word'):
            words_ms.append(W['Ms'].sum())
            word_mistakes.append(W['Mistake'].sum())
            word_lens.append(len(W))

        # This will automatically check that 
        #   the number of words from the text (words list) matches the number of words found in C (other lists)
        # W = pd.DataFrame({'Attempt':word_keystrokes, 'Keystrokes':word_strokes})
        self.words = pd.DataFrame({'Word':words, 'Ms':words_ms, 'Mistakes':word_mistakes, 'Length':word_lens})


    # Parse the second half of typingLog
    #   This contains *all* the keystrokes
    def parse_entries(self):
        # Do this to make regex better (and remove extra comma at end of typingLog)
        T = ',' + self._tl[1][:-1]


        ## WINDOWS
        # Split into windows
        # This makes it so regex matches are all unique (no overlaps)
        wordnums_pattern_ = _PATTERN_WORDNUMS.replace('(', '').replace(')', '')

        # Each window starts with "\d,\d"
        text_inds, num_strokes = zip(*re.findall(_PATTERN_WORDNUMS, T))
        text_inds = [int(i) for i in text_inds]
        num_strokes = [int(i) for i in num_strokes]

        # Get the (string) keystrokes per window
        windows = re.split(wordnums_pattern_, T)[1:]
        # * Implicity checks for equality of length (windows processed correctly)
        self.windows = pd.DataFrame({'Window': windows, 'NumStrokes': num_strokes, 'TextInd': text_inds})


        ## KEYSTROKES / ENTRIES
        # For each keystroke
        strokes, strokes_ms, strokes_window = ([] for _ in range(3))
        # For each entry
        window_inds, chars, ops, char_ms, window_nums, stroke_nums, stroke_inds = \
            ([] for _ in range(7))
        
        # Each window now comprises of multiple keystrokes
        # Each keystroke is comprised of one or more entries
        # Each entry has this pattern
        #   ,{ms},{kestroke},

        # This is which stroke (from the strokes DataFrame) the entry(ies) come from
        stroke_num = -1
        # for w in W.itertuples():
        for w in range(len(windows)):
            # For regex robustness
            window_ = ',' + windows[w]
            # Use this pattern to separate out each keystroke
            #   Looking for the ms is more robust than finding keystrokes
            ms_vals = re.findall(_PATTERN_MS, window_)
            keystrokes = re.split(_PATTERN_MS, window_)[1:]

            # Window contains multiple keystrokes
            for stroke_ms_, stroke in zip(ms_vals, keystrokes):
                stroke_num += 1
                # This is the window that the stroke came from
                window_num = w
                
                # Each entry has this pattern
                #   ,({ind}{op}{char})+,
                entries = re.findall(_PATTERN_ENTRIES, stroke)
                num_entries = len(entries)
                assert(num_entries > 0)

                stroke_ms = int(stroke_ms_[1:-1])
                # NOTE: Two ways of saving ms for keystrokes w multiple entries
                entries_ms = np.full(num_entries, stroke_ms)
                # 1. First entry gets the whole ms, other entries get 0
                entries_ms[1:] = 0
                # 2. Divide between number of entries
                # TODO: Change this when satisfied with TypingLog parsing
                # entries_ms = entries_ms / num_entries

                for i, (char_ind_, op, char) in enumerate(entries):
                    # The "character" found with regex can be \" (two characters), so take the last index
                    chars.append(char)
                    char_ms.append(entries_ms[i])
                    ops.append(op)

                    stroke_nums.append(stroke_num)
                    # Individual entry's index in the stroke
                    stroke_inds.append(i)

                    window_nums.append(window_num)
                    # This is "purportly" the index within the full window
                    # ! This is from typingLog, so can be buggy!
                    window_inds.append(int(char_ind_))

                strokes.append(stroke)
                strokes_ms.append(stroke_ms)
                strokes_window.append(w)

        self.strokes = pd.DataFrame({'Stroke': strokes, 'Ms': strokes_ms, 'Window': strokes_window})
        self.entries = pd.DataFrame({'Window': window_nums, 'WindowInd': window_inds, 'Char': chars, 'Op': ops, 'Ms': char_ms, 'Stroke': stroke_nums, 'StrokeInd': stroke_inds})
        return self.entries, self.strokes, self.windows


    # Parse the first half of typingLog
    # This contains all the correct characters and the time it took to *correctly* type each
    #   * not all entries, only correct ones shown
    def parse_chars(self) -> pd.DataFrame:
        T =  self._tl[0]
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
        self._chars = pd.DataFrame({'Char':chars, 'Ms':ms})
        return self._chars


    # From the typingLog (first half, with characters), generate the original text
    def generate_text(self):
        if not self.chars.empty:
            DF = self.chars
        elif not self._chars.empty:
            DF = self._chars
        else:
            self.parse_chars()
            DF = self._chars
        return ''.join(DF['Char'])


    # FOR DEBUG ONLY: Recreate the original typingLog string using the DataFrames (_char and words)
    def regenerate_typinglog(self):
        if self.words.empty:
            self.generate()
        if self._chars.empty:
            self.parse_chars()

        # Just rearrange columns to make string generation easier
        W = pd.DataFrame({
            'TextInd': self.words['TextInd'],
            'NumStrokes': self.words['NumStrokes'],
            'Window': self.words['Window'],
        })
        # SECOND HALF
        tl1 = ','.join(W.astype(str).values.flatten()) + ','

        # Special function to convert characters for first half of typinglog
        #   \b is prepended to certain characters
        #   Perform unicode conversion if needed
        def convert_char_tl0(c):
            if c.isdigit() or c == '-':
                return '\\b' + c 
            else:
                return TypingLog.reverse_char_clean(c)
        # FIRST HALF
        tl0 = ''.join(self._chars['Char'].apply(convert_char_tl0) + self._chars['Ms'].apply(str))
        tl1 = ''.join([TypingLog.reverse_char_clean(c) for c in tl1])

        return self.header + ',' + tl0 + '|' + tl1


    # Purely to remove unnecessary dataframes to reduce memory usage
    def clear_data(self, dfs='all'):
        if dfs == 'all':
            dfs = ['entries', 'strokes', 'windows', '_chars', 'chars', 'words']
        elif dfs == 'unnecessary':
            dfs = ['entries', 'strokes', 'windows', '_chars', 'words']
        
        if isinstance(dfs, str):
            dfs = [dfs]

        for df in dfs:
            setattr(self, df, pd.DataFrame())



    # Reconstruct text from typingLog entries DataFrame (containing keystrokes)
    # Returns final reconstructed text, as well as all intermediate text after each keystroke
    #   Length of states array should be the same as the number of keystrokes
    # NOTE: The generate() class method uses this function AFTER altering the entries DataFrame
    #       This affects the states list (though not the final typed text)
    @staticmethod
    def generate_typed_input(TL: pd.DataFrame) -> tuple[str, list[str]]:
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
        return text, states
    


    # The typingLog contains some keystrokes consisting of multiple entries
    # Some such keystrokes are just multiple additions in a row
    # To make typingLog parsing eeasier, split these into multiple keystrokes
    @staticmethod
    def split_addition_entries(TL):
        # For speed, use numpy arrays
        strokes = TL['Stroke'].to_numpy()
        stroke_inds = TL['StrokeInd'].to_numpy()
        ops = TL['Op'].to_numpy()

        # When we split keystrokes, the keystroke numbers will increase for the remaining keystrokes
        # Keep track of that increase
        stroke_offset = 0
        # Compute indices upfront, since we will be editing the strokes array
        stroke_inds_all = [strokes == s for s in range(strokes[-1]+1)]
        for inds in stroke_inds_all:
            strokes[inds] += stroke_offset
            # Number of entries
            ne = np.count_nonzero(inds)
            # Normal keystroke (w one entry), ignore
            if ne == 1:
                continue

            if np.all(ops[inds] == '+'):
                # Split keystroke
                strokes[inds] = strokes[inds] + np.arange(ne)
                stroke_offset += (ne-1)

                stroke_inds[inds] = 0

        TL['Stroke'] = strokes
        TL['StrokeInd'] = stroke_inds
        return TL
    

    # Case: User duoble-taps punctuation at the end of text
    # The text was correctly typed, but extra punctuation was typed 
    #   at the end and saved as part of the same keystroke
        # Check if there was a duplicate punctuation at the end of the text (TypeRacer bug, or input lag perhaps)
        # The duplicate characters should be from the same keystroke (probably)
    # Case: User inputs any incorrect character at the end after the text was already completed 
    @staticmethod
    def adjust_end_entries(TL0, TL1):
        TL = TL1.copy()

        # Find the number of duplicate characters at the end of a string
        num_dups = lambda text: len(text) - len(text.rstrip(text[-1]))
        
        # Final keystroke
        #   may contain multiple character entries
        end_stroke = TL.iloc[-1]['Stroke']
        inds_end_stroke = TL['Stroke'] == end_stroke
        end_chars = ''.join( TL[inds_end_stroke]['Char'] )

        # TRUTH
        typed_text = ''.join(TL0['Char'])
        end_char = typed_text[-1]
        
        # Find the true number of duplicate characters at the end
        # Limit the number of duplicates to the length of the stroke
        dups = min(num_dups(typed_text), len(end_chars))

        # From entries: find the purported end character
        ind_end = end_chars.rindex(end_char)
        chars_extra = len(end_chars) - ind_end - 1
        # Get the number of duplicate characters at 
        dups_extra = num_dups(end_chars[:ind_end+1]) - dups

        assert(dups_extra >= 0), 'ERROR: adjust_end_entries: dups_extra < 0.\n' + \
                                 'Means the entries DataFrame does not contains all the existing characters'

        if rows_extra := chars_extra + dups_extra == 0:
            return TL
        
        stroke_ms = TL[inds_end_stroke]['Ms'].to_numpy()
        ms_extra = stroke_ms[-rows_extra:].sum()

        # Only update the non-zero elements
        stroke_ms_ = stroke_ms[:-rows_extra]
        inds_nonzero = stroke_ms_ != 0
        # Allocate the extra time to these rows
        stroke_ms_[inds_nonzero] += ms_extra / np.count_nonzero(inds_nonzero)

        return TL[:len(TL)-(chars_extra+dups_extra)].copy()


    # The typingLog contains two parts, separated by a pipe character
    # This function returns the index of that character
    # To account for the possiblity of a pipe in the text itself, search for a specific regex pattern
    #   i.e. pipe character followed by digits (first word index, should always be 0)
    @staticmethod
    def typinglog_pipe(tl) -> int:
        pipe_pattern = r'\|\d+,'
        pipe_occurence = list(re.finditer(pipe_pattern, tl))

        return pipe_occurence[-1].start()
    

    # The typingLog contains two formats, separated by a pipe character
    # To account for the possiblity of a pipe in the text itself, search for a specific regex pattern
    #   i.e. pipe character followed by digits (first word index, should always be 0)
    @staticmethod
    def clean_typinglog(tl: str) -> str:
        tl = tl.replace('\\\\', '\\')
        tl = tl.replace('\\"', '"')

        unicode_pattern = '\\\\u[0-9a-fA-F]{4}'
        unicode_occurences = re.finditer(unicode_pattern, tl)
        for unicode_occurence in unicode_occurences:
            unicode_str = unicode_occurence.group()
            tl = tl.replace(unicode_str, unicode_str.encode('utf-8').decode('unicode-escape'))

        return tl


    # Input must be a single character
    # Converts to the "escaped" version found in the origianl typingLog (extracted from TypeRacer)
    @staticmethod
    def reverse_char_clean(c: str) -> str:
        assert(len(c) == 1)
        if c == '\\':
            return '\\\\'
        elif c == '"':
            return '\\"'
        else:
            # Converts unicode characters to the "string" version (\u00__)
            # For almost all characters, this should do nothing
            return c.encode('unicode-escape').decode('utf-8').replace('\\x', '\\u00')
