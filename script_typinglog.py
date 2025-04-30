# You should have run script_html already

#%%
# Text with numbers (years) and parentheses
ind = 7264
# ind = 7263


#%%
tl = races.loc[ind, 'TypingLog']
text = texts.loc[races.loc[ind, 'TextID'], 'Text']

#%%
# Split by |
T = tl.split('|')[0][12:]

ms = []
chars = []

specialChar = '\\b'

def checkForSpecialChar(T, i):
    # Look for \\b, return 
    if T[i] == specialChar[0]:
        # i+4 (instead of i+3) because we're counting the character that should be in front of \\b
        if i+(len(specialChar)+1) <= len(T):
            if T[i:i+len(specialChar)] == specialChar:
                return True
        
    return False

inSpecialChar = False
indSpecialChar = -1
newDigit = True

for i in range(len(T)):
    c = T[i]

    if not inSpecialChar:
        inSpecialChar = checkForSpecialChar(T, i)

    if inSpecialChar:
        indSpecialChar += 1
        if indSpecialChar == len(specialChar):
            inSpecialChar = False
            indSpecialChar = -1
            chars.append(c)

        continue

    # print(i)

    # First check if there is a special character
    if T[i].isdigit():
        if newDigit:
            ms.append(c)
            newDigit = False
        else:
            ms[-1] += c
    else:
        chars.append(c)
        newDigit = True

    
    if c == specialChar[0]:
        raise Exception('Should not be here')



assert(''.join(chars) == text)



#%%

soup = BeautifulSoup(read_url(wp_race.format('goldrik', ind), useSelenium=True), 'html.parser')
mistakes, section_texts, section_wpms = mistakes_sections_from_soup(soup)


#%%

sectionTextTest = ' '.join(section_texts) == text
if sectionTextTest:
    print('Section text matches exactly')
else:
    print('Section text does not match exactly')

    section_texts_ = section_texts.copy()
    section_texts_[-1] += text[-1]
    sectionTextTest_ = ' '.join(section_texts_) == text

    print(f'\tAfter adding last character "{text[-1]}" to last section:', end=' ')
    if sectionTextTest_:
        print('Section text matches')
    else:
        print('Section text does not match')


#%%
## Determine Sections

num_sections = 8

section_len_chars = len(text) / num_sections
# Exact characters where the text may be divided into sections
inds = (np.arange(num_sections+1) * section_len_chars).astype(int)

start_index = 0
section_texts_ = []
# for ind_start,ind_end in zip(inds[:-1], inds[1:]):
for i in range(num_sections):
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
    section_texts_.append(text[ind_start:ind_end].strip())

section_texts_[-1] = section_texts_[-1][:-1]
_ = [(print(s0), print(s1)) for s0,s1 in zip(section_texts, section_texts_)]
[section0 == section1 for section0, section1 in zip(section_texts, section_texts_)]

