# Aurik Sarker
# 09 May 2025

import pandas as pd
import pickle
import os

from typeracer_utils import *
from parse_soup import *
from url_formatstrings import url_base, url_user, url_races, url_race, url_text

_profile_attrs = [
    'Name', 'StartDate', 'Location', 'Keyboard', 'Races', 'AverageWPM', 'BestWPM', 'Avatar'
]
_race_attrs = {
    'DateTime': object, 'WPM': int, 'Accuracy': float, 'Points': int, 'Rank': int, 'NumRacers': int, 'Opponents': object,
    'TextID': int, 'TypedText': str, 'TypingLog': str,
}
_racer_attrs = [
    'Racers', 'WPMs', 'Accuracies', 'TypingLogs'
]
_text_attrs = {
    'Text': str, 'Title': str, 'Type': str, 'Author': str, 'Submitter': str,
    'WPM': int, 'Accuracy': float, 'NumWords': int, 'NumChars': int, 'Races': object,
}


class TypeRacerUser:
    """TypeRacerUser"""

    # Denotes Guest racer
    # Use this instead of the string literal 'Guest', since that is a valid username
    GUEST = ''

    def __init__(self, user, fn_htmls: str=None):
        self.user = user

        self.htmls = {}
        self.htmls_pkl = fn_htmls

        # For debugging: May want to reload htmls from TypeRacer (in case of site update)
        self.reload = False

        self.clear()
        self.load_htmls()


    def clear(self):
        self.profile = {attr:None for attr in _profile_attrs}
        self.races = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in _race_attrs.items()})
        self.racers = pd.DataFrame({col: pd.Series(dtype=object) for col in _racer_attrs})
        self.texts = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in _text_attrs.items()})

        # This stores the opponent information for each race (Names, Races, Ranks)
        # Used by populate_racers() to build opponents' race URLs without reloading the user's race
        # Private vars used because the user will get this info through the racers DataFrame
        self._opponents = {}

    
    def update(self):
        self.populate_profile()
        self.populate_races()
        self.populate_racers()
        self.populate_texts()

    
    def load_htmls(self, htmls_pkl=None):
        if htmls_pkl is None:
            htmls_pkl = self.htmls_pkl

        if htmls_pkl is None or not os.path.exists(htmls_pkl):
            htmls_ = {}
        else:
            with open(htmls_pkl, 'rb') as f:
                htmls_ = pickle.load(f)
        
        # Combine dictionaries
        self.htmls = {**self.htmls, **htmls_}
    

    def populate_races(self, num_load=1e12, inds_load=None):
        # First, determine the full list of races we want to load 
        if inds_load is None:
            inds_load = range(self.profile['Races'], 0, -1)

        races_load = get_missing_indices(self.races, inds_load)
        races_load = races_load[:num_load]

        races_dict = {c:[] for c in self.races.columns if c != 'Points'}
        races_inds = []
        for race in races_load:
            # LOAD PAGE
            wp = url_race.format(self.user, race)
            try:
                _, soup = read_url(wp, self.htmls, reloadHtml=self.reload)

                # TEXT ID
                textID = extract_textid(soup)

                # EVERYTHING ELSE (race details and opponents)
                R, O = parse_race(soup)
                tl = extract_typing_log(soup)

            except:
                continue

            races_dict['DateTime'].append(R['Date'])
            races_dict['WPM'].append(R['Speed'])
            races_dict['Accuracy'].append(R['Accuracy'])
            # races_dict['Points'].append()
            races_dict['Rank'].append(R['Rank'])
            races_dict['NumRacers'].append(R['NumRacers'])
            races_dict['Racers'].append(O['Users'])
            races_dict['TextID'].append(textID)

            self._opponents[race] = {'Names': O['Users'], 'Races': O['Races'], 'Ranks': O['Ranks']}
            
            if tl:
                TL_ = parse_typinglog_simple(tl)
                typedText = ''.join(TL_['Char'])
            else:
                typedText = ''

            races_dict['TypedText'].append(typedText)
            races_dict['TypingLog'].append(tl)

            races_inds.append(race)
        
        if len(races_inds):
            races_ = pd.DataFrame(races_dict, index=races_inds)

            self.races = pd.concat([self.races, races_])
            self.races = adjust_dataframe_index(self.races)
    

    # This is dependent on the races dataframe
    # Uses _opponent_... private vars first, but can pull from race webpage if needed
    def populate_racers(self):
        # First, determine which races need to be loaded
        racers_load = get_missing_indices(self.racers, self.races)

        racers_dict = {c:[] for c in self.racers.columns}
        racers_inds = []
        for race in racers_load:
            try:
                players = self.extract_racers(race)
            except:
                continue

            racers_dict['Racers'].append( players['Users'] )
            racers_dict['WPMs'].append( players['WPMs'] )
            racers_dict['Accuracies'].append( players['Accs'] )
            racers_dict['TypingLogs'].append( players['TLS'] )

            racers_inds.append(race)
        
        if len(racers_inds):
            racers_ = pd.DataFrame(racers_dict, index=racers_inds)

            self.racers = pd.concat([self.racers, racers_])
            self.racers = adjust_dataframe_index(self.racers)
    

    # For a single race
    def extract_racers(self, race: int) -> dict:
        # For all racers in a given race (except Guests), get the
        #   username, wpm, accuracy, typingLog

        # Assuming the race was already loaded
        if race in self.races.index:
            row = self.races.loc[race]
            details = {
                'NumRacers': row['NumRacers'],
                'Speed': row['WPM'],
                'Accuracy': row['Accuracy'],
                'Rank': row['Rank'],
                'TypingLog': row['TypingLog'],
            }

            opponents = {
                'Users': self._opponents[race]['Names'],
                'Races': self._opponents[race]['Races'],
                'Ranks': self._opponents[race]['Ranks'],
            }

        else:
            soup = read_url(url_race.format(self.user, race), self.htmls, reloadHtml=self.reload)[1]
            details, opponents = parse_race(soup)


        # Number of racers
        N = details['NumRacers']

        # Initialize racers information
        players = {
            'Users': [TypeRacerUser.GUEST] * N,
            'WPMs': [-1] * N,
            'Accs': [np.nan] * N,
            'TLs': [''] * N
        }

        # All human players: self + opponents
        players = [self.user] + opponents['Users']
        races = [race] + opponents['Races']

        for player, race in zip(players, races):
            if player == self.user:
                player_details = details
            else:
                soup = read_url(url_race.format(player, race), self.htmls, reloadHtml=self.reload)[1]
                player_details = parse_race(soup)[0]
                player_details['TypingLog'] = extract_typing_log(soup)
            
            rank = details['Rank']
            players['Users'][rank-1] = player
            players['WPMs'][rank-1] = details['Speed']
            players['Accs'][rank-1] = details['Accuracy']
            players['TLs'][rank-1] = details['TypingLog']

        return players


    # For each textID in the races dataframe, 
    # Populate the texts dataframe
    #   avoids re-reading text if already found in texts
    def populate_texts(self):
        texts_dict = {c:[] for c in self.texts.columns if c != 'Races'}
        textIDs = []
        for textID in self.races['TextID']:
            # Check if we've loaded this text before
            if (textID not in self.texts.index) and (textID not in textIDs):
                try:
                    _, soup = read_url(url_text.format(textID), self.htmls)
                    T = parse_text(soup)
                except:
                    continue

                texts_dict['Text'].append(T['text'])
                texts_dict['Title'].append(T['title'])
                texts_dict['Type'].append(T['type'])
                texts_dict['Author'].append(T['author'])
                texts_dict['Submitter'].append(T['submitter'])
                texts_dict['WPM'].append(T['wpm'])
                texts_dict['Accuracy'].append(T['accuracy'])

                texts_dict['NumWords'].append( len(T['text'].split(' ')) )
                texts_dict['NumChars'].append( len(T['text']) )

                textIDs.append(int(textID))

        if len(textIDs):
            texts_ = pd.DataFrame(texts_dict, index=textIDs)
            self.texts = pd.concat([self.texts, texts_])

            self.texts = adjust_dataframe_index(self.texts, sortDesc=False)

        # Update races (for each text)
        for textID in self.texts.index:
            self.texts.at[textID, 'Races'] = self.races[self.races['TextID'] == textID].index.tolist()


    def method1(self, arg1):
        """
        A brief description of method1.

        Input:
            arg1 (type): Description of arg1.

        Ouptut:
            type: Description of the return value.
        """
        # Method implementation
        pass
