from sympy import symbols, Matrix
import numpy as np
import scipy.linalg as sl
from math import sqrt, factorial



"""
A1: matriz de puntuaciones en la primera practica de tiros.
A1(i,j): puntuación del tiro j-esimo sobre la i-esima diana
"""

s = symbols('s')

simb_matrix = Matrix([
[1,  0,  0, 0,  0,  0,  0, 0,  0,  0,   0,   0],
[0, s,  0, 0,  0,  0,  0, 0,  0,  0,   0,   0],
[1,  1, s, 0,  0,  0,  0, 0,  0,  0,   0,   0],
[0,  0,  0, 1, -1,  2, -1, 0,  0,  0,   0,   0],
[0,  0,  0, 0,  1,  0,  0, 0,  0,  0,   0,   0],
[0,  0,  0, 0,  0, -1,  1, 0,  0,  0,   0,   0],
[0,  0,  0, 0,  0,  0,  1, 0,  0,  0,   0,   0],
[0,  0,  0, 0,  0,  0,  0, 1, s,  0,   0,   0],
[0,  0,  0, 0,  0,  0,  0, 0,  1, s,   0,   0],
[0,  0,  0, 0,  0,  0,  0, 0,  0,  1, s,   0],
[0,  0,  0, 0,  0,  0,  0, 0,  0,  0,   1, s],
[0,  0,  0, 0,  0,  0,  0, 0,  0,  0,   0,   1]])

punt_diana = 5
evalf_matrix = simb_matrix.evalf(subs={s:punt_diana})

num_matrix = np.array(evalf_matrix, dtype=np.float)
