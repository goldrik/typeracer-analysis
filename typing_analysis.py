# Aurik Sarker
# 3 May 2025

# Parse and analyse the typingLog data
#   javascript variable extracted from the race webpage
# typingLog parsing may differ from TypeRacer's javascript methods

# Includes other functions for parsing the text

import numpy as np
import pandas as pd
# from datetime import datetime, timedelta

##
# TEXT

# Determine the sections
def text_to_sections(text: str, numSections: int=8) -> list[str]:
    section_len_chars = len(text) / numSections
    # Exact characters where the text may be divided into sections
    inds = (np.arange(numSections+1) * section_len_chars).astype(int)

    start_index = 0
    sections = []
    # for ind_start,ind_end in zip(inds[:-1], inds[1:]):
    for i in range(numSections):
        ind_start = inds[i]
        ind_end = inds[i+1]

        # If the end_index is not at a space and not at the end of the text, adjust
        if ind_end < len(text) and not text[ind_end].isspace():
            # Move the end_index to the previous or next space depending on where the divide is in the word
            if ind_end - ind_start > len(text[ind_end:].split()[0]):
                while ind_end > ind_start and not text[ind_end-1].isspace():
                    ind_end -= 1
            else:
                while ind_end < len(text) and not text[ind_end].isspace():
                    ind_end += 1

        inds[i+1] = ind_end
        sections.append(text[ind_start:ind_end].strip())

    sections[-1] = sections[-1][:-1]

    return sections

##
# TYPINGLOG
