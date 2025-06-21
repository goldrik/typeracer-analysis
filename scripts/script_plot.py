# Aurik Sarker
# 21 June 2025

#%%
#!%load_ext autoreload
#!%autoreload 2

import numpy as np
import pandas as pd

import os
import dotenv
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

from parse_soup import *
from typeracer_utils import *
from typing_analysis import *
from url_formatstrings import *

from TypeRacerUser import TypeRacerUser
from TypingLog import TypingLog
from Keyboard import Keyboard


#%%
## ENV VARS
dotenv.load_dotenv()
USER: str = os.getenv('USER')
FH_PKL: str = os.getenv('FH_PKL')

FN_PKL_USER = os.path.join(FH_PKL, f'typeracer_{USER}.pkl')
FN_PKL_HTMLS = os.path.join(FH_PKL, 'typeracer_htmls.pkl')


#%%
## LOAD
print('Loading TypeRacerUser...')
userdata = TypeRacerUser.load(FN_PKL_USER)


print('Generating TypingLogs...')
# Generate TypingLog output for each race
TLs=  {}
for race in userdata.races.index:
    print(f'\tRace {race}', end='\r')
    TLs[race] = TypingLog(userdata.races.at[race, 'TypingLog'])
    TLs[race].generate()
print()


#%%
## 
races_to_numpy = lambda c: userdata.races[c].to_numpy()[::-1]


#%%
## WPM: TypeRacer
fig, ax = plt.subplots()

# This is the plot that Typeracer USED TO SHOW for average WPM over time
# Mean of each 100 races

wpms = userdata.races['WPM'].to_numpy()[::-1]
wpms_range = range(0, len(wpms), 100)

wpms_avg = [round(np.mean(wpms[i:i+100])) for i in wpms_range]

ax.plot(wpms_range, wpms_avg, '.-')

ax.set_xlabel('Race')
ax.set_ylabel('WPM')
ax.set_title('TypeRacer Average WPM')


#%%
## KEYBOARD ACCURACY

K = Keyboard(data='Accuracy')
chars_counts = {}
chars_mistakes = {}

for TL in TLs.values():
    for char,mistake in zip(TL.chars['Char'], TL.chars['Mistake']):
        chars_counts[char] = chars_counts.get(char, 0) + 1
        chars_mistakes[char] = chars_mistakes.get(char, 0) + int(mistake)

chars_accs = {char:(1 - chars_mistakes[char] / chars_counts[char]) * 100 for char in chars_counts.keys()}

for char in chars_counts.keys():
    K.keys[char].value = chars_accs[char]
    # K.keys[char].value = np.log(chars_counts[char])


# K.plot()
K.plot(cm_range = (92,98))

sorted_chars_accs = sorted(chars_accs.items(), key=lambda x: x[1], reverse=True)
for char, acc in sorted_chars_accs:
    print(f'Char: {char}, Accuracy: {acc:.2f}%, Count: {chars_counts[char]}')


#%%
## KEYBOARD TIME

K = Keyboard(data='Time per Key (ms)')
chars_ms_total = {}

for TL in TLs.values():
    for char,ms in zip(TL.chars['Char'], TL.chars['Ms']):
        chars_ms_total[char] = chars_ms_total.get(char, 0) + ms

chars_ms = {char:(chars_ms_total[char] / chars_counts[char]) for char in chars_ms_total.keys()}

for char in chars_ms.keys():
    K.keys[char].value = chars_ms[char]
    # K.keys[char].value = np.log(chars_ms[char])


# K.plot()
K.plot(cm_range = (75, 200))

sorted_chars_ms = sorted(chars_ms.items(), key=lambda x: x[1])
for char, ms in sorted_chars_ms:
    print(f'Char: {char}, Time: {ms:.2f} ms, Count: {chars_counts[char]}')


#%%
## KEYBOARD MISTAKE CHARS

K = Keyboard(data='Accuracy')
chars_typed = {}

for TL in TLs.values():
    for char,typed in zip(TL.chars['Char'], TL.chars['Typed']):
        if char not in chars_typed:
            chars_typed[char] = {}
        chars_typed[char][typed] = chars_typed[char].get(typed, 0) + 1

for char in chars_typed.keys():
    chars_typed[char].pop(char, None)

chars_typed_most = {char:max(chars_typed[char], key=chars_typed[char].get) for char in chars_typed.keys() if chars_typed[char]}

fig, axs = K.plot(plot_values=False)

for key in chars_typed_most.keys():
    shift = int( K.keys[key].shift )

    arrow_start = K.keys[key].loc
    arrow_end = K.keys[chars_typed_most[key]].loc
    if arrow_start != arrow_end:
        # Draw an arrow
        axs[shift].annotate('', xytext=arrow_start, xy=arrow_end,
            arrowprops=dict(arrowstyle='->', color='red'))

    else:
        axs[shift].plot(arrow_start[0], arrow_start[1], 'o', color='red')


#%%
##
binned = userdata.races.groupby(pd.Grouper(key='DateTime', freq='ME', label='left'))['WPM']

means = binned.mean()
stds = binned.std()
errs = binned.agg(lambda group: group.std() / np.sqrt(len(group)))

# dts = [interval.left for interval in means.index.values]
dts = means.index.values
plt.errorbar(dts, means.values, yerr=stds.values, fmt='.')


#%%
fig, ax = plt.subplots()

# sns.scatterplot(data=userdata.races, x='WPM', y='Accuracy', hue='DateTime', ax=ax, s=20)
# ax.legend_.remove()

wpms = races_to_numpy('WPM')
accs = races_to_numpy('Accuracy')

dts = races_to_numpy('DateTime').astype('float')
dts = (dts - dts.min()) / (dts.max() - dts.min())
colors = [plt.cm.jet(d) for d in dts]

ax.scatter(accs, wpms, c=colors, s=8, alpha=0.25)



#%%
fig, axs = plt.subplots(2, 2, figsize=(8,6))

# Initialize data matrix (N x 2)
A = np.stack([userdata.texts['NumChars'].to_numpy(), userdata.texts['WPM'].to_numpy()]).T

gmm = GaussianMixture(n_components=2).fit(A)
userdata.texts['Group'] = gmm.predict(A)

# Plot the two groups
for g,c in enumerate(['b', 'r']):
    inds = userdata.texts['Group'] == g
    axs[0,0].scatter(userdata.texts[inds]['NumChars'], userdata.texts[inds]['WPM'], c=c, s=8, alpha=0.25)
    axs[1,0].scatter(userdata.texts[inds]['NumChars'], userdata.texts[inds]['Accuracy'], c=c, s=8, alpha=0.25)
    axs[0,1].scatter(userdata.texts[inds]['Accuracy'], userdata.texts[inds]['WPM'], c=c, s=8, alpha=0.25)

# Now try on user races
userdata.races['Group'] = range(len(userdata.races))
for i, row in userdata.texts.iterrows():
    for race in row['Races']:
        userdata.races.at[race, 'Group'] = row['Group']

binned = userdata.races.groupby(pd.Grouper(key='DateTime', freq='ME', label='left'))['Group']
sizes = binned.size()
means = binned.mean()

# dts = [interval.left for interval in means.index.values]
dts = means.index.values
axs[1,1].scatter(dts, means.values, s=sizes)

axs[1,0].set_xlabel('Num Chars')
axs[0,0].set_ylabel('Average WPM')
axs[1,0].set_ylabel('Average Accuracy')

axs[0,1].set_xlabel('Average Accuracy')
axs[0,1].set_ylabel('Average WPM')

axs[1,1].set_xlabel('Date (binned)')
axs[1,1].set_ylabel('% Races in Group 1, per bin')

fig.suptitle('Group TypeRacer Texts: Examine apparent bimodal distribution of texts\nEach datapoint is a text (except for Axes 1,1)')
axs[0,0].set_title('Texts: WPM vs Num Chars')
axs[1,0].set_title('Texts: WPM vs Accuracy')
axs[0,1].set_title('Races: WPM vs Accuracy')
axs[1,1].set_title('Races: Texts Served to User')


axs[0,0].legend(['Text Group 0', 'Text Group 1'], fontsize=10)


fig.tight_layout()




