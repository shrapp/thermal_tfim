from functions import calc_data
import numpy as np
import logging

logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

Ns = [50,100]

taus = np.logspace(-5, 0, 25)
ws = [0.0, 0.1, 0.05, 0.2, 0.5, 1.0, 5.0, 0.01]
calc_data(Ns, taus, ws)