# Aurik Sarker
# 09 May 2025

import pandas as pd
import pickle
import os

from typeracer_utils import *
from parse_soup import *
from TypingLog import TypingLog
from url_formatstrings import url_base, url_user, url_races, url_race, url_text

_profile_attrs = [
    'Name', 'StartDate', 'Location', 'Keyboard', 'Races', 'AverageWPM', 'BestWPM', 'Avatar'
]
_race_attrs = {
    'DateTime': object, 'WPM': int, 'Accuracy': float, 'Rank': int, 'NumRacers': int, 'Opponents': object,
    'TextID': int, 'TypedText': str, 'TypingLog': str,
}
_racer_attrs = [
    'Racers', 'WPMs', 'Accuracies', 'TypingLogs'
]
_text_attrs = {
    'Text': str, 'Title': str, 'Type': str, 'Author': str, 'Submitter': str,
    'WPM': int, 'Accuracy': float, 'NumWords': int, 'NumChars': int, 'Races': object,
}
_results_attrs = {
    'Date': object, 'WPM': int, 'Accuracy': float, 'Points': int, 'Rank': int, 'NumRacers': int,
}


class TypeRacerUser:
    """TypeRacerUser"""

    # Denotes Guest racer
    # Use this instead of the string literal 'Guest', since that is a valid username
    GUEST = ''
    # Default "max races"
    # Just some number far beyond the max of races for a user
    # MAX_RACES = int(1e12)

    def __init__(self, user, fn_htmls: str=None):
        self.clear()

        self.htmls_pkl = fn_htmls
        self.load_htmls()

        self.user = user
        self.populate_profile()


    def clear(self):
        self.htmls = {}
        self.htmls_pkl = None
        # For debugging: May want to reload htmls from TypeRacer (in case of site update)
        self.reload = False

        # Profile is separate from clear_data()
        # It should always be set
        self.profile = {attr:None for attr in _profile_attrs}
        self.clear_data()


    def clear_data(self):
        self.races = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in _race_attrs.items()})
        self.racers = pd.DataFrame({col: pd.Series(dtype=object) for col in _racer_attrs})
        self.texts = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in _text_attrs.items()})
        self.results = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in _results_attrs.items()})

        # This stores the opponent information for each race (Names, Races, Ranks)
        # Used by populate_racers() to build opponents' race URLs without reloading the user's race
        # Private vars used because the user will get this info through the racers DataFrame
        self._opponents = {}

    
    def update(self, num_races=1e12):
        self.populate_profile()
        self.populate_races(num_load=num_races)
        self.populate_racers()
        self.populate_texts()
        if self.htmls_pkl is not None:
            self.save_htmls()

    
    def load_htmls(self, htmls_pkl=None):
        if htmls_pkl is None:
            htmls_pkl = self.htmls_pkl

        if htmls_pkl is None or not os.path.exists(htmls_pkl):
            print('WARNING: htmls were not loaded, htmls_pkl class variable either not set or does not exist')
            htmls_ = {}
        else:
            with open(htmls_pkl, 'rb') as f:
                htmls_ = pickle.load(f)
        
        # Combine dictionaries
        self.htmls = {**self.htmls, **htmls_}

    
    def save_htmls(self, htmls_pkl=None):
        if htmls_pkl is None:
            htmls_pkl = self.htmls_pkl

        if htmls_pkl is not None:
            with open(htmls_pkl, 'wb') as f:
                pickle.dump(self.htmls, f)
        else: 
            print('WARNING: htmls were not saved, htmls_pkl class variable not set')
    

    def populate_profile(self):
        # Do not use htmls dict here, this should always be reloaded
        url = url_user.format(self.user)
        soup = read_url(url)[1]
        
        info = parse_profile_info(soup)
        stats = parse_profile_stats(soup)

        self.profile['Name'] = info['Name']
        self.profile['StartDate'] = info['Racing Since']
        self.profile['Location'] = info['Location']
        self.profile['Keyboard'] = info['Keyboard']
        self.profile['Avatar'] = info['Avatar']

        self.profile['Races'] = stats['Races']
        self.profile['AverageWPM'] = stats['Full Avg.']
        self.profile['BestWPM'] = stats['Best Race']
    

    def populate_races(self, num_load=1e12, inds_load=None):
        # First, determine the full list of races we want to load 
        if inds_load is None:
            inds_load = range(self.profile['Races'], 0, -1)

        races_load = get_missing_indices(self.races, inds_load)
        races_load = races_load[:int(num_load)]

        races_dict = {c:[] for c in self.races.columns}
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
            races_dict['Rank'].append(R['Rank'])
            races_dict['NumRacers'].append(R['NumRacers'])
            races_dict['Opponents'].append(O['Users'])
            races_dict['TextID'].append(textID)

            self._opponents[race] = {'Names': O['Users'], 'Races': O['Races'], 'Ranks': O['Ranks']}
            
            if tl:
                typedText = TypingLog(tl).generate_text()
            else:
                typedText = ''

            races_dict['TypedText'].append(typedText)
            races_dict['TypingLog'].append(tl)

            races_inds.append(race)
        
        if len(races_inds):
            races_ = pd.DataFrame(races_dict, index=races_inds)

            if len(self.races):
                self.races = pd.concat([self.races, races_])
            else:
                self.races = races_
            self.races = adjust_dataframe_index(self.races)

        return self.races
    

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
            racers_dict['TypingLogs'].append( players['TLs'] )

            racers_inds.append(race)
        
        if len(racers_inds):
            racers_ = pd.DataFrame(racers_dict, index=racers_inds)

            self.racers = pd.concat([self.racers, racers_])
            self.racers = adjust_dataframe_index(self.racers)

        return self.racers
    

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
        players_details = {
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
            
            rank = player_details['Rank']
            players_details['Users'][rank-1] = player
            players_details['WPMs'][rank-1] = int(player_details['Speed'])
            players_details['Accs'][rank-1] = float(player_details['Accuracy'])
            players_details['TLs'][rank-1] = player_details['TypingLog']

        return players_details


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

        return self.texts


    def populate_results(self, num_load=None):
        # First, get the total number of races
        indsPrev = self.results.index

        n_races = self.profile['Races']
        racesMissing = get_missing_indices(self.results, n_races)

        if num_load is None:
            num_load = n_races
        racesToLoad = racesMissing[:num_load]

        # This is what the dataframe indices should be afterwards
        indsAfter = np.array(list(set(indsPrev) | set(racesToLoad)))

        wp = self._get_next_results_url( racesToLoad, n_races)

        while wp != '': 
            try:
                _, soup = read_url(wp)
            except:
                print(f'\t...failed to read webpage. Stop loading races')
                break
            
            results_ = parse_results(soup)
            results_ = results_.rename(columns={'Speed':'WPM', 'Place':'Rank'})
            self.results = pd.concat([self.results, results_])

            wp = self._get_next_results_url(racesToLoad, 
                n_races, lastRaceLoaded=results_.index.min(), currentPageSoup=soup)

        self.results = self.results.loc[indsAfter].copy()
        # TODO This prints "0 duplicate rows" even when no results loaded
        self.results = adjust_dataframe_index(self.results)

        return self.results


    # Given the list of loaded races and list of races to load
    #   return the URL to load the next set of missing races
    # lastRaceLoaded and currentPageSoup are used handle the "older results" link
    #   both or neither variable must be set at once
    def _get_next_results_url(self, raceInds:int, maxRaces:int, numRacesToLoad:int=100, 
                        lastRaceLoaded:int=None, currentPageSoup:BeautifulSoup=None) -> str:
        str_older = 'load older results'

        # Gaps in races dataframe which may need to be filled in
        inds = get_missing_indices(self.results, raceInds)
        if not inds.size:
            # All races loaded -> return no url
            return ''
        
        ind = inds[0]
        # If the latest race is missing, start from the beginning
        if ind == maxRaces:
            return url_races.format(self.user, numRacesToLoad, '')
        
        # Check if this index matches the final index in the recently loaded races
        if (ind+1) == lastRaceLoaded:
            if currentPageSoup is None:
                raise Exception('ERROR: currentPageSoup must be set if lastRaceLoaded is input')
            older_div = currentPageSoup.find('a', 
                                                string=lambda text: str_older in text.lower())
            if older_div is None:
                raise Exception('ERROR: Could not find "load older results" link')
            
            return url_base + 'race_history' + older_div['href']
        
        # Otherwise, start from the date (right after) the missing race
        search_date = next_day_to_str(self.results.loc[ind+1, 'Date'])
        return url_races.format(self.user, numRacesToLoad, search_date)


    def save(self, obj_pkl):
        # If we have an htmls pkl file, save those to a separate file, 
        #   not as part of the TypeRacerUser obj
        # Otherwise, just save everthing together
        save_htmls = self.htmls_pkl is None

        if not save_htmls:
            self.save_htmls()
            self.htmls = {}
        
        with open(obj_pkl, 'wb') as f:
            pickle.dump(self, f)

        if not save_htmls:
            self.load_htmls() 


    @staticmethod
    def load(obj_pkl):
        if os.path.exists(obj_pkl):
            with open(obj_pkl, 'rb') as f:
                obj = pickle.load(f)
        else:
            raise Exception(f'ERROR: {obj_pkl} file does not exist')
        
        obj.load_htmls()
        return obj
        
