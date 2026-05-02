import numpy as np
from parameters import *
# Функция плотности
def rho(p_val):
    return ro0 * (1 + cf * (p_val - p0_ref))