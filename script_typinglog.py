# You should have run script_html already


#%%
def tl1_to_char_ms(tl):
    # Split by |
    T = tl.split('|')[0]
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
        
    return chars, ms


#%%
# Text with numbers (years) and parentheses
ind = 7264
# With double quotes
# ind = 7498

inds = np.random.choice(races.index, 300, replace=False)
inds = [ind]

    #%%
for ind in inds:
    tl = races.loc[ind, 'TypingLog']
    text = texts.loc[races.loc[ind, 'TextID'], 'Text']

    # chars, ms = tl1_to_char_ms(tl)
    # assert(''.join(chars) == text)


    # Split by |
    T = tl.split('|')[1]
    T = T.split(',', 2)[-1]

    tt = T.split(',')
    tt0 = tt[::2]
    tt1 = tt[1::2]

    [print(t0, ' ', t1) for t0, t1 in zip(tt0, tt1)]

    # print('success')


