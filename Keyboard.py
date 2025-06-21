# Aurik Sarker
# 30 May 2025

# import pandas as pd

# from typeracer_utils import *
# from parse_soup import *

from dataclasses import dataclass
from enum import Enum

import math
import random
import numpy as np
import matplotlib.pyplot as plt

#%%


# Stores every key in the keyboard
#   Location (in coordinates)
#     x (column), y (row)
#   Finger
#     0 - pinky
#     1 - ring
#     2 - middle
#     3 - index
#     4 - thumb
#   Hand
#     0 - left
#     1 - right
#   Shift

# ! Only QWERTY for now


class Keyboard:
    """Keyboard"""

    def __init__(self, keyboard='qwerty', data='Value'):
        self.keyboard = keyboard.lower()

        if self.keyboard == 'qwerty':
            keys = [
                # TODO find text with all keys and shift keys, take video of myself for this
                Key('`',   (0.00, 4.), Finger.PINKY,  Hand.LEFT,  False),
                Key('1',   (1.00, 4.), Finger.RING,   Hand.LEFT,  False),
                Key('2',   (2.00, 4.), Finger.RING,   Hand.LEFT,  False),
                Key('3',   (3.00, 4.), Finger.MIDDLE, Hand.LEFT,  False),
                Key('4',   (4.00, 4.), Finger.MIDDLE, Hand.LEFT,  False),
                Key('5',   (5.00, 4.), Finger.MIDDLE, Hand.LEFT,  False),
                Key('6',   (6.00, 4.), Finger.INDEX,  Hand.RIGHT, False),
                Key('7',   (7.00, 4.), Finger.MIDDLE, Hand.RIGHT, False),
                Key('8',   (8.00, 4.), Finger.MIDDLE, Hand.RIGHT, False),
                Key('9',   (9.00, 4.), Finger.RING,   Hand.RIGHT, False),
                Key('0',  (10.00, 4.), Finger.PINKY,  Hand.RIGHT, False),
                Key('-',  (11.00, 4.), Finger.PINKY,  Hand.RIGHT, False),
                Key('=',  (12.00, 4.), Finger.PINKY,  Hand.RIGHT, False),

                Key('q',   (1.50, 3.), Finger.PINKY,  Hand.LEFT,  False),
                Key('w',   (2.50, 3.), Finger.RING,   Hand.LEFT,  False),
                Key('e',   (3.50, 3.), Finger.MIDDLE, Hand.LEFT,  False),
                Key('r',   (4.50, 3.), Finger.INDEX,  Hand.LEFT,  False),
                Key('t',   (5.50, 3.), Finger.INDEX,  Hand.LEFT,  False),
                Key('y',   (6.50, 3.), Finger.INDEX,  Hand.RIGHT, False),
                Key('u',   (7.50, 3.), Finger.INDEX,  Hand.RIGHT, False),
                Key('i',   (8.50, 3.), Finger.MIDDLE, Hand.RIGHT, False),
                Key('o',   (9.50, 3.), Finger.RING,   Hand.RIGHT, False),
                Key('p',  (10.50, 3.), Finger.PINKY,  Hand.RIGHT, False),
                Key('[',  (11.50, 3.), Finger.MIDDLE, Hand.RIGHT, False),
                Key(']',  (12.50, 3.), Finger.RING,   Hand.RIGHT, False),
                Key('\\', (13.50, 3.), Finger.RING,   Hand.RIGHT, False),

                Key('a',   (1.83, 2.), Finger.PINKY,  Hand.LEFT,  False),
                Key('s',   (2.83, 2.), Finger.RING,   Hand.LEFT,  False),
                Key('d',   (3.83, 2.), Finger.MIDDLE, Hand.LEFT,  False),
                Key('f',   (4.83, 2.), Finger.INDEX,  Hand.LEFT,  False),
                Key('g',   (5.83, 2.), Finger.INDEX,  Hand.LEFT,  False),
                Key('h',   (6.83, 2.), Finger.INDEX,  Hand.RIGHT, False),
                Key('j',   (7.83, 2.), Finger.INDEX,  Hand.RIGHT, False),
                Key('k',   (8.83, 2.), Finger.MIDDLE, Hand.RIGHT, False),
                Key('l',   (9.83, 2.), Finger.RING,   Hand.RIGHT, False),
                Key(';',  (10.83, 2.), Finger.PINKY,  Hand.RIGHT, False),
                Key('\'', (11.83, 2.), Finger.PINKY,  Hand.RIGHT, False),

                Key('z',    (2.17, 1.), Finger.PINKY,  Hand.LEFT,  False),
                Key('x',    (3.17, 1.), Finger.RING,   Hand.LEFT,  False),
                Key('c',    (4.17, 1.), Finger.INDEX,  Hand.LEFT,  False),
                Key('v',    (5.17, 1.), Finger.INDEX,  Hand.LEFT,  False),
                Key('b',    (6.17, 1.), Finger.INDEX,  Hand.LEFT,  False),
                Key('n',    (7.17, 1.), Finger.INDEX,  Hand.RIGHT, False),
                Key('m',    (8.17, 1.), Finger.INDEX,  Hand.RIGHT, False),
                Key(',',    (9.17, 1.), Finger.MIDDLE, Hand.RIGHT, False),
                Key('.',   (10.17, 1.), Finger.PINKY,  Hand.RIGHT, False),
                Key('/',   (11.17, 1.), Finger.PINKY,  Hand.RIGHT, False),


                # SHIFT

                Key('~',   (0.00, 4.), Finger.MIDDLE, Hand.LEFT,  True), 
                Key('!',   (1.00, 4.), Finger.MIDDLE, Hand.LEFT,  True),
                Key('@',   (2.00, 4.), Finger.RING,   Hand.LEFT,  True),
                Key('#',   (3.00, 4.), Finger.MIDDLE, Hand.LEFT,  True),
                Key('$',   (4.00, 4.), Finger.MIDDLE, Hand.LEFT,  True),
                Key('%',   (5.00, 4.), Finger.INDEX,  Hand.LEFT,  True),
                Key('^',   (6.00, 4.), Finger.INDEX,  Hand.LEFT,  True),
                Key('&',   (7.00, 4.), Finger.MIDDLE, Hand.RIGHT, True),
                Key('*',   (8.00, 4.), Finger.MIDDLE, Hand.RIGHT, True),
                Key('(',   (9.00, 4.), Finger.MIDDLE, Hand.RIGHT, True),
                Key(')',  (10.00, 4.), Finger.MIDDLE, Hand.RIGHT, True),
                Key('_',  (11.00, 4.), Finger.MIDDLE, Hand.RIGHT, True),
                Key('+',  (12.00, 4.), Finger.MIDDLE, Hand.RIGHT, True),

                Key('Q',   (1.50, 3.), Finger.RING,   Hand.LEFT,  True),
                Key('W',   (2.50, 3.), Finger.RING,   Hand.LEFT,  True),
                Key('E',   (3.50, 3.), Finger.MIDDLE, Hand.LEFT,  True),
                Key('R',   (4.50, 3.), Finger.INDEX,  Hand.LEFT,  True),
                Key('T',   (5.50, 3.), Finger.INDEX,  Hand.LEFT,  True),
                Key('Y',   (6.50, 3.), Finger.INDEX,  Hand.RIGHT, True),
                Key('U',   (7.50, 3.), Finger.INDEX,  Hand.RIGHT, True),
                Key('I',   (8.50, 3.), Finger.MIDDLE, Hand.RIGHT, True),
                Key('O',   (9.50, 3.), Finger.RING,   Hand.RIGHT, True),
                Key('P',  (10.50, 3.), Finger.PINKY,  Hand.RIGHT, True),
                Key('{',  (11.50, 3.), Finger.MIDDLE, Hand.RIGHT, True),
                Key('}',  (12.50, 3.), Finger.RING,   Hand.RIGHT, True),
                Key('|',  (13.50, 3.), Finger.RING,   Hand.RIGHT, True),

                Key('A',   (1.83, 2.), Finger.RING,   Hand.LEFT,  True),
                Key('S',   (2.83, 2.), Finger.RING,   Hand.LEFT,  True),
                Key('D',   (3.83, 2.), Finger.MIDDLE, Hand.LEFT,  True),
                Key('F',   (4.83, 2.), Finger.INDEX,  Hand.LEFT,  True),
                Key('G',   (5.83, 2.), Finger.INDEX,  Hand.LEFT,  True),
                Key('H',   (6.83, 2.), Finger.INDEX,  Hand.RIGHT, True),
                Key('J',   (7.83, 2.), Finger.INDEX,  Hand.RIGHT, True),
                Key('K',   (8.83, 2.), Finger.MIDDLE, Hand.RIGHT, True),
                Key('L',   (9.83, 2.), Finger.RING,   Hand.RIGHT, True),
                Key(':',  (10.83, 2.), Finger.PINKY,  Hand.RIGHT, True),
                Key('"',  (11.83, 2.), Finger.PINKY,  Hand.RIGHT, True),

                Key('Z',   (2.17, 1.), Finger.RING,   Hand.LEFT,  True),
                Key('X',   (3.17, 1.), Finger.RING,   Hand.LEFT,  True),
                Key('C',   (4.17, 1.), Finger.INDEX,  Hand.LEFT,  True),
                Key('V',   (5.17, 1.), Finger.INDEX,  Hand.LEFT,  True),
                Key('B',   (6.17, 1.), Finger.INDEX,  Hand.LEFT,  True),
                Key('N',   (7.17, 1.), Finger.INDEX,  Hand.RIGHT, True),
                Key('M',   (8.17, 1.), Finger.INDEX,  Hand.RIGHT, True),
                Key('<',   (9.17, 1.), Finger.MIDDLE, Hand.RIGHT, True),
                Key('>',  (10.17, 1.), Finger.RING,   Hand.RIGHT, True),
                Key('?',  (11.17, 1.), Finger.PINKY,  Hand.RIGHT, True),

                Key(' ',    (7.00, 0.), Finger.THUMB,  Hand.RIGHT, False),
                Key('shift',(0.00, 1.), Finger.PINKY,  Hand.LEFT,  False),
                Key('ctrl', (1.00, 0.), Finger.PINKY,  Hand.LEFT,  False),
                Key('\b',  (13.00, 4.), Finger.MIDDLE, Hand.RIGHT, False)
            ]
        
        elif self.keyboard == 'dvorak':
            pass
        else:
            raise Exception(f'ERROR: Keyboard {self.keyboard} not supported')

        self.keys = {k.key:k for k in keys}

        # For data
        self.data = data


    def plot(self, plot_values=True, cm_range=None):
        fig, axs = plt.subplots(2,1)
        [ax.set_xlim(-1, 14) for ax in axs]
        [ax.set_ylim(-1, 5) for ax in axs]
        [ax.set_xticks([]) for ax in axs]
        [ax.set_yticks([]) for ax in axs]
        [ax.set_aspect('equal') for ax in axs]

        keys = self.keys.values()


        # ! This gets reversed after adding colorbar
        # Doing _adjust() after the colorbar doesnt work either
        fig.subplots_adjust(hspace=0.05)


        # Plot the data
        if plot_values:
            # Get all the values
            vals = np.array([key.value for key in keys])
            # Only plot if any non nan values exist
            if not np.isnan(vals).all():
                # Get the min and max values
                _vmin = np.nanmin(vals)
                _vmax = np.nanmax(vals)

                if cm_range is None:
                    vmin = 0
                    vmax = 1
                    if _vmin < 0 or _vmax > 1:
                        vmin = _vmin
                        vmax = _vmax
                else:
                    vmin = cm_range[0]
                    vmax = cm_range[1]
                
                for key in keys:
                    val = key.value
                    if not np.isnan(val):
                        # Normalize value between 0 and 1
                        nval = (val-vmin)/(vmax-vmin)
                        nval = min(max(nval, 0), 1)
                        color = plt.cm.jet( float(nval) ) [:-1]

                        # Add a kind of blurring/ripple effect
                        _s = [50, 200, 400]
                        _a = [0.25, 0.20, 0.10]
                        # _s = [50]
                        # _a = [1.0]
                        [axs[ int(key.shift) ].scatter(key.loc[0], key.loc[1], s=s, color=color, alpha=a) for s,a in zip(_s, _a)]

                # Colorbar
                sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin, vmax))
                cbar = fig.colorbar(sm, ax=axs, shrink=0.80)
                cbar.set_label(self.data, fontsize=12, fontweight='bold', rotation=270, x=-15)
                cbar.set_ticks([vmin, vmax])


        # Plot the keys (as text)
        for key in keys:
            _key = key.key

            # Which axes to plot
            _axs = [int(key.shift)]
            fw = 'bold'
            if _key in ['ctrl', 'shift', ' ', '\b']:
                _axs = [0,1]
                fw = 'normal'
            
            if _key == ' ':
                _key = 'space'
            if _key == '\b':
                _key = 'del'

            [axs[_a].text(key.loc[0], key.loc[1], _key, fontweight=fw, ha='center', va='center') for _a in _axs]


        # fig.tight_layout()
        return fig, axs


class Finger(Enum):
    PINKY = 0
    RING = 1
    MIDDLE = 2
    INDEX = 3
    THUMB = 4

class Hand(Enum):
    LEFT = 0
    RIGHT = 1


@dataclass
class Key:
    """Key"""

    key: str
    loc: tuple
    finger: Finger
    hand: Hand
    shift: bool
    # For storing data
    value: float = np.nan

    @property
    def row(self):
        return int(self.loc[1])
    @property
    def home(self):
        return self.row == 2
    
    # Absolute Distance, as well as row and col dist
    def dist(self, key):
        return math.sqrt((self.loc[0] - key.loc[0])**2 + (self.loc[1] - key.loc[1])**2), \
               int(self.loc[0] - key.loc[0]), int(self.loc[1] - key.loc[1])


