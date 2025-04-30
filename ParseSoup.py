# Aurik Sarker
# 30 April 2025

# Contains modules for parsing TypeRacer pages
# The pages must already have been read in (to html string) and procesed using BeautifulSoup
#   i.e. BeautifulSoup(html, 'html.parser')

# The inputs are all BeautifulSoup objects
# The outputs may be
#   DataFrame - for profile page (table of races)
#   tuple - for pages corresponding to one single race or text

import numpy as np
import pandas as pd