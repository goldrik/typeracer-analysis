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

    # text = text[:-1]

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


# Parse the first half of typinglog
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
    TL = pd.DataFrame({'char':chars, 'ms':ms})

    return TL

