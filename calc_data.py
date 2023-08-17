import numpy as np

from functions import calc_data

Ns = [50,100]

# taus = np.logspace(-5, 0, 25)
# ws = [0.0, 0.1, 0.05, 0.2, 0.5, 1.0, 5.0, 0.01]

# calc_data(Ns, taus, ws)

calc_data(Ns, [0.12345], [0.12345])