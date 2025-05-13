import unittest
import numpy as np
from TypeRacerUser import TypeRacerUser
from TypingLog import TypingLog

# All checks from script_typeraceruser
# Check if number of races = number of racers
# Check if all texts accounted for

class TestTypeRacerUser(unittest.TestCase):
    obj = TypeRacerUser('goldrik')
    # Last loaded race
    race = None

    def test_loaded(self):
        # Check that the race number is in the racers dataframe
        self.assertIn(self.race, self.obj.races.index, 'Race not loaded')
        self.assertIn(self.race, self.obj.racers.index, 'Race opponents not loaded')

        textID = self.race.loc[race]['TextID']
        self.assertIn(textID, self.obj.texts.index, f'Race\'s, correponding TextID {textID} not loaded')

class TestTypingLog(unittest.TestCase):
    TL = None
    # Last loaded race
    text = None

    def test_loaded(self):
        # Check that the race number is in the racers dataframe
        self.assertIn(self.race, self.obj.races.index, 'Race not loaded')
        self.assertIn(self.race, self.obj.racers.index, 'Race opponents not loaded')

        textID = self.race.loc[race]['TextID']
        self.assertIn(textID, self.obj.texts.index, f'Race\'s, correponding TextID {textID} not loaded')



# === Main loop: create object + run tests repeatedly ===
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    obj = TestTypeRacerUser.obj

    # Get the number of races
    n_races = obj.profile['Races']
    
    # races = np.arange(n_races,0,-1)    
    races = np.random.choice(np.arange(n_races,0,-1), 10, replace=False)
    
    # Create a fresh object and assign to test class
    for race in races:
        print('\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n')
        print('Loading race', race, 'into TypeRacerUser object')
        obj.update(races=[race])
        TestTypeRacerUser.race = race

        print('Generating TypingLog')
        tl = obj.races.loc[race, 'TypingLog']
        TestTypingLog.text = obj.races.loc[race, 'TypedText']
        TestTypingLog.TL = TypingLog(tl)

        # Load and run the tests
        suite1 = unittest.defaultTestLoader.loadTestsFromTestCase(TestTypeRacerUser)
        suite2 = unittest.defaultTestLoader.loadTestsFromTestCase(TestTypingLog)

        runner.run(suite1)
        runner.run(suite2)

