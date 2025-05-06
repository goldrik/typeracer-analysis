# You should have run script_html already

import re

#%%


def tl1_to_char_ms(tl:str, text:str):
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
    for w in range(len(ww0)):
        W = ',' + ww1[w]
        tt0 = re.findall(ms_pattern, W)
        tt1 = re.split(ms_pattern, W)[1:]

        word_strokes.append(int(ww0[w][1]))
        for t0, t1 in zip(tt0, tt1):
            ms = int(t0[1:-1])

            strs = re.findall(char_pattern, t1)
            if len(strs) != 0:
                i += 1
            for word_ind, op, char in strs:
                ms_all.append(ms)

                word_ind_ = int(word_ind)
                c = char[-1]
                word_inds.append(word_ind_)
                ops.append(op)
                chars.append(c)

                pd_inds.append(i)
                word_nums.append(w)

                ms = 0


    TL = pd.DataFrame({'word':word_nums, 'ind':word_inds, 'char':chars, 'ms':ms_all, 'op':ops}, index=pd_inds)

    return TL, word_strokes

def tl_df_to_words(TL, text):
    T = len(TL)


    words=  text.split(' ')
    words[:-1] = [w+' ' for w in words[:-1]]

    chars_text = []
    chars_ms = []
    mistakes = []
    chars_typed = []

    t = ''
    word = []
    w = 0
    w_i = 0
    ms_ = 0
    mistake = False
    mistake_char = ''
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

        # Check each time the word reaches correct status
        if ''.join(word) == words[w][:w_i+1]:
        # if word == words[w][:len(word)]:
            c_ = words[w][w_i]
            
            chars_text.append(c_)
            ms_ += TL.iloc[i]['ms']
            chars_ms.append(ms_)
            mistakes.append(mistake)
            if not mistake: 
                chars_typed.append(c_)
            else:
                chars_typed.append(mistake_char)

            w_i += 1
            ms_ = 0
            mistake = False
            mistake_char = ''
        else:
            mistake = True
            if mistake_char == '':
                mistake_char = c
            ms_ += TL.iloc[i]['ms']


        if (i == T-1) or ((c == ' ') and (TL.iloc[i+1]['ind'] == 0) and (TL.iloc[i+1]['op'] == '+')):
            t += ''.join(word)
            word = []

            w += 1
            w_i = 0
            ms_ = 0
            mistake = False
            mistake_char = ''

    C = pd.DataFrame({'char':chars_text, 'ms':chars_ms, 'mistake':mistakes, 'typed':chars_typed})

    words_ms = []
    word_lens = []
    word_strokes = []

    word_inds = TL['word'].to_numpy()
    assert(len(np.unique(word_inds)) == len(words))
    for i in range(len(words)):
        inds = word_inds == i
        words_ms.append(TL['ms'][inds].sum())
        word_lens.append(len(words[i]))

        inds = TL[inds].index
        word_strokes.append(inds.max() - inds.min() + 1)

    W = pd.DataFrame({'word':words, 'ms':words_ms, 'len':word_lens, 'strokes':word_strokes})


    return t, C, W


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
# ind = 7541

# ind = 7511

# ind = 7548

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
    assert(''.join(TL0['Char']) == text)


    # TL1, _ = tl1_to_char_ms(tl, text)
    # TL1 = parse_typinglog_complete(tl)
    TL1, C,W, text_ = parse_typinglog(tl, text)
    _, word_strokes = parse_typinglog_wordvals(tl)


    # [print(f'{a}\t{b}\t{c}\t{d}') for a,b,c,d in 
    #     zip(TL1['ind'], TL1['op'], TL1['char'], TL1['ms'])]
    

    wpm0 = (chars_total / 5) / (TL0['Ms'].sum() / 1e3 / 60)
    wpm1 = (chars_total / 5) / (TL1['Ms'].sum() / 1e3 / 60)
    assert(wpm0 == wpm1)

    # print(f'{wpm} {wpm1:.2f}')

    assert(TL1['WordInd'].max()+1 == words_total)
    assert(sum(word_strokes) == TL1.index[-1]+1)

    # text_ = tl_df_to_text(TL1)
    # text_, C, W = tl_df_to_words(TL1, text)
    

    assert(TL1['Ms'].sum() == C['Ms'].sum())
    assert(TL1['Ms'].sum() == W['Ms'].sum()) 



    assert(text_ == text)
    assert(''.join(C['Char']) == text)
    assert(''.join(W['Word']) == text)

    acc_ = 100 * (1 - C['Mistake'].sum() / len(C))
    # print(f'{acc_:.2f}\t{acc:.2f}')

    assert(all(np.array(word_strokes) == W['Keystrokes'].to_numpy()))


