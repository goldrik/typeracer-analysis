# You should have run script_html already

import re

#%%
def split_tl(tl):
    pipe_pattern = r'\|\d+,'
    pipe_occurence = list(re.finditer(pipe_pattern, tl))
    return pipe_occurence[-1].start()


def tl0_to_char_ms(tl):
    # Split by |
    T = tl[:split_tl(tl)]
    T = T.split(',', 3)[-1]

    ms = []
    chars = []

    inSpecialChar = False
    lookingForMs = False

    for c in T:
        # # ASSUMING NO TEXT HAS THE \ CHARACTER

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


def tl1_to_char_ms(tl):
    # Split by |
    T = tl[split_tl(tl)+1:]
    T = T.split(',', 2)[-1]

    # Do this to make regex better
    T = ',' + T[:-1]

    # FORMAT
    # ,{ms},{ind+char}+,
    
    # The {ms} portion is easy to handle
    # Find those using regex and split the string
    ms_pattern = r',-?\d+,'
    char_pattern = r'\d+[+\-$].'
    char_pattern_groups = r'(\d+)([+\-$])(\\"|.)'

    tt0 = re.findall(ms_pattern, T)
    tt1 = re.split(ms_pattern, T)[1:]

    ms_all = []
    word_inds = []
    ops = []
    chars = []

    i = -1
    pd_inds = []
    for t0, t1 in zip(tt0, tt1):
        ms = int(t0[1:-1])

        strs = re.findall(char_pattern_groups, t1)
        if len(strs) != 0:
            i += 1
        for word_ind, op, char in strs:
            ms_all.append(ms)
            ms = 0

            word_inds.append(int(word_ind))
            ops.append(op)
            chars.append(char[-1])

            pd_inds.append(i)

    TL = pd.DataFrame({'char':chars, 'ms':ms_all, 'op':ops, 'ind':word_inds}, index=pd_inds)

    return TL


def tl1_to_words(tl):
    # Split by |
    T = tl[split_tl(tl)+1:]
    T = ',' + T

    wordnums_pattern = r',(\d+),(\d+),'

    ww0 = re.findall(wordnums_pattern, T)

    word_inds, word_strokes = zip(*ww0)
    word_inds = [int(i) for i in word_inds]
    word_strokes = [int(s) for s in word_strokes]
    return word_inds, word_strokes


#%%
# Text with numbers (years) and parentheses
ind = 7264
# With double quotes
# ind = 7498

# iive vs live
ind = 7541

ind = 7511


#%%

inds = races.index
# inds = np.random.choice(races.index, 300, replace=False)
# inds = [ind]

for race in inds:
    textID = races.loc[race, 'TextID']

    tl = races.loc[race, 'TypingLog']
    text = texts.loc[textID, 'Text']
    wpm = races.loc[race, 'WPM']
    acc = races.loc[race, 'Accuracy']
    # numwords = texts.loc[textID, 'NumWords']

    chars_total = len(text)

    if tl.count('|') != 1:
        print('Warning: Erroneous number of pipes found')

    TL0 = tl0_to_char_ms(tl)
    assert(''.join(TL0['char']) == text)


    TL1 = tl1_to_char_ms(tl)


    # [print(f'{a}\t{b}\t{c}\t{d}') for a,b,c,d in 
    #     zip(TL1['ind'], TL1['op'], TL1['char'], TL1['ms'])]
    

    wpm0 = (chars_total / 5) / (TL0['ms'].sum() / 1e3 / 60)
    wpm1 = (chars_total / 5) / (TL1['ms'].sum() / 1e3 / 60)
    assert(wpm0 == wpm1)

    print(f'{wpm} {wpm1:.2f}')

    word_inds, word_strokes = tl1_to_words(tl)

    T = len(TL1)
    t = ''
    word = []
    for i in range(T):
        c = TL1.iloc[i]['char']
        word_ind = TL1.iloc[i]['ind']
        op = TL1.iloc[i]['op']

        if op == '+':
            word.append(c)
            assert(word[word_ind] == c)

        elif op == '-':
            assert(word[word_ind] == c)
            word.pop(word_ind)

        elif op == '$':
            word[word_ind] = c
            assert(word[word_ind] == c)

        if (i == T-1) or ((c == ' ') and (TL1.iloc[i+1]['ind'] == 0) and (TL1.iloc[i+1]['op'] == '+')):
            t += ''.join(word)
            word = []
        

    # words = []
    # word_strokes_ = word_strokes.copy()
    # newWord = True
    # for c in TL1['char']:
    #     if newWord:
    #         words.append(c)
    #         newWord = False
    #     else:
    #         words[-1] += c

    #     word_strokes_[0] -= 1
    #     if word_strokes_[0] == 0:
    #         word_strokes_.pop(0)
    #         newWord = True

    # t = ''
    # for i in range(len(TL1)):
    #     if TL1['op'].iloc[i] == '+':
    #         t += TL1['char'].iloc[i]
    #     elif TL1['op'].iloc[i] == '-':
    #         # t = t[:-1]
    #         for j in range(len(t)-1, -1, -1):
    #             if t[j] == TL1['char'].iloc[i]:
    #                 t = t[:j] + t[j+1:]
    #                 break
        # else:
        #     t += chars[i] * ms_all[i]

    # chars_str = ''.join(TL1['char'])[::-1]
    # inds_list = TL1['ind'].to_list()[::-1]
    # ops_str = ''.join(TL1['op'])[::-1]

    # I = len(TL1)
    # i = 0
    # t = ''

    # newWord = False
    # spaceInds = []
    # while True:
    #     t += chars_str[i]
    #     for wordInd in range(inds_list[i]-1, -1,-1):
    #         # Find the next index which matches wordInd
    #         i = inds_list.index(wordInd, i)
    #         t += chars_str[i]
    #     i = chars_str.find(' ', i)
    #     if i == -1:
    #         break

    #     spaceInds.append(i)

    # t = t[::-1]

            
            
        # TL1.iloc[i]['char']

    # # print(t)
    assert(t == text)


