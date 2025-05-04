# Aurik Sarker
# 03 May 2025

# Format strings for building a TypeRacer URL
# Usage: 
#   url_user.format('goldrik') -> 'https://data.typeracer.com/pit/profile?user=goldrik'

url_base = 'https://data.typeracer.com/pit/'
# User page, contains user num races, average wpm, etc
url_user = url_base + 'profile?user={}'
# Races, contains table with N races and their basic details
url_races = url_base + 'race_history?user={}&n={}&startDate={}'
# Single race, contains detailed information for each race, 
#   including opponents and typingLog variable
url_race = url_base + 'result?id=|tr:{}|{}'
# Text, details about single text, no user information
url_text = url_base + 'text_info?id={}'
