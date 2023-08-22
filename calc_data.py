import logging
import pandas as pd
import numpy as np

from functions import calc_data

data = pd.read_csv('data.csv')


Ns = data['N'].unique()

taus = data['tau'].unique()
noises = data['noise'].unique()

calc_data(Ns, taus, noises)