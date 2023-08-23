import logging

import numpy as np
import pandas as pd

from functions import calc_data

data = pd.read_csv('data.csv')


Ns = data['N'].unique()

taus = data['tau'].unique()
noises = data['noise'].unique()

# # remove taus greater than 10
# taus = taus[taus <= 100]

# remove noises greater than 10
noises = noises[noises <= 2]


print(Ns)
print(taus)
print(noises)

calc_data(Ns, taus, noises)