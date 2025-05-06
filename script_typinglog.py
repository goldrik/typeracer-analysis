# You should have run script_html already

import re

#%%


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

    chars_total = texts.loc[textID, 'NumChars']
    words_total = texts.loc[textID, 'NumWords']

    # chars_total = len(text)
    # words_total = len(text.split(' '))

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
    

    assert(TL1['Ms'].sum() == C['Ms'].sum())
    assert(TL1['Ms'].sum() == W['Ms'].sum()) 


    assert(text_ == text)
    assert(''.join(C['Char']) == text)
    assert(''.join(W['Word']) == text)

    acc_ = 100 * (1 - C['Mistake'].sum() / len(C))
    inds_chars = (C['Char'] != ' ').to_numpy()
    acc__ = 100 * (1 - C.loc[inds_chars, 'Mistake'].sum() / np.count_nonzero(inds_chars))
    print(f'{acc:.2f}\t{acc_:.2f}\t{acc__:.2f}')

    assert(all(np.array(word_strokes) == W['Keystrokes'].to_numpy()))

    mistake_words = W['Mistake'].to_numpy()
    assert(all(W.loc[~mistake_words, 'Word'] == W.loc[~mistake_words, 'Attempt']))

