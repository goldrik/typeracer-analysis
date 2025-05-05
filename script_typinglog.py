# You should have run script_html already

import re

#%%

def tl1_to_char_ms(tl, text):
    # Split by |
    T = tl[typinglog_pipe(tl)+1:]
    # Do this to make regex better
    T = ',' + T[:-1]

    # Split into words
    wordnums_pattern = r',(\d+),(\d+),'
    wordnums_pattern_ = wordnums_pattern.replace('(', '').replace(')', '')
    # The {ms} portion is easy to handle
    # Find those using regex and split the string
    ms_pattern = r',-?\d+,'
    char_pattern = r'(\d+)([+\-$])(\\"|.)'

    # FORMAT
    # ,{ms},{ind+char}+,
    
    ww0 = re.findall(wordnums_pattern, T)
    ww1 = re.split(wordnums_pattern_, T)[1:]

    ms_all = []
    word_inds = []
    ops = []
    chars = []

    word_nums = []
    pd_inds = []

    word_strokes = []

    words = text.split(' ')
    assert(len(words) == len(ww0))

    i = -1
    w_ = 0
    for w in range(len(ww0)):
        W = ',' + ww1[w]
        tt0 = re.findall(ms_pattern, W)
        tt1 = re.split(ms_pattern, W)[1:]

        word = words[w]

        word_strokes.append(int(ww0[w][1]))
        for t0, t1 in zip(tt0, tt1):
            ms = int(t0[1:-1])

            strs = re.findall(char_pattern, t1)
            if len(strs) != 0:
                i += 1
            for word_ind, op, char in strs:
                ms_all.append(ms)
                ms = 0

                word_ind_ = int(word_ind)
                word_inds.append(word_ind_)
                ops.append(op)
                chars.append(char[-1])

                pd_inds.append(i)
                word_nums.append(w)

                # if word[w] == 

    TL = pd.DataFrame({'word':word_nums, 'ind':word_inds, 'char':chars, 'ms':ms_all, 'op':ops}, index=pd_inds)
    
    return TL, word_strokes


def tl_df_to_text(TL):
    T = len(TL)
    t = ''
    word = []
    for i in range(T):
        c = TL.iloc[i]['char']
        word_ind = TL.iloc[i]['ind']
        op = TL.iloc[i]['op']

        if op == '+':
            word.append(c)
            assert(word[word_ind] == c)

        elif op == '-':
            assert(word[word_ind] == c)
            word.pop(word_ind)

        elif op == '$':
            word[word_ind] = c
            assert(word[word_ind] == c)

        if (i == T-1) or ((c == ' ') and (TL.iloc[i+1]['ind'] == 0) and (TL.iloc[i+1]['op'] == '+')):
            t += ''.join(word)
            word = []

    return t


#%%
# Text with numbers (years) and parentheses
ind = 7264
# With double quotes
# ind = 7498

# iive vs live
ind = 7541

# ind = 7511


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
    words_total = len(text.split(' '))

    assert(tl.count('|') == 1)
    assert(tl[typinglog_pipe(tl)+1] == '0')

    TL0 = parse_typinglog_simple(tl)
    assert(''.join(TL0['char']) == text)


    TL1, word_strokes = tl1_to_char_ms(tl, text)


    # [print(f'{a}\t{b}\t{c}\t{d}') for a,b,c,d in 
    #     zip(TL1['ind'], TL1['op'], TL1['char'], TL1['ms'])]
    

    wpm0 = (chars_total / 5) / (TL0['ms'].sum() / 1e3 / 60)
    wpm1 = (chars_total / 5) / (TL1['ms'].sum() / 1e3 / 60)
    assert(wpm0 == wpm1)

    # print(f'{wpm} {wpm1:.2f}')

    assert(TL1['word'].max()+1 == words_total)
    assert(sum(word_strokes) == TL1.index[-1]+1)

    text_ = tl_df_to_text(TL1)

    assert(text_ == text)


