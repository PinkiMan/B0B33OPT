__author__ = "Pinkas Matěj"
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__ = "0.1.0"
__maintainer__ = "Pinkas Matěj"
__email__ = "pinkas.matej@gmail.com"
__status__ = "Prototype"
__date__ = "20/04/2025"
__created__ = "18/04/2025"

"""
Filename: lp.py
"""

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt


def vyhra(c, k):
    """
    :param c: [c1, c10, c0, c02, c2]
    :param k: total money to be bet
    :return: x
    """

    c_obj = [0, 0, 0, 0, 0, -1]

    A = [
        [-c[0], -c[1], 0, 0, 0, 1],         # -1.27x1 -1.02x10 + t <= 0
        [0, -c[1], -c[2], -c[3], 0, 1],     # -1.02x10 -4.70x0 -3.09x02 + t <= 0
        [0, 0, 0, -c[3], -c[4], 1]          # -3.09x02 -9.00x2 + t <= 0
    ]
    b = [0, 0, 0]

    A_eq = [[1, 1, 1, 1, 1, 0]]
    b_eq = [k]

    bounds = [(0, None)] * 5 + [(None, None)]

    res = linprog(c=c_obj, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    return res.x[:-1]


def vyhra2(c, k, m):
    """
    :param c: [c1, c0, c2]
    :param k: total money
    :param m: minimum bet per event
    :return: x
    """

    c_obj = [0, 0, 0, -1]

    A = [
        [-c[0], 0, 0, 1],   # -c1 x1 + t <= 0
        [0, -c[1], 0, 1],   # -c0 x0 + t <= 0
        [0, 0, -c[2], 1]    # -c2 x2 + t <= 0
    ]
    b = [0, 0, 0]

    A_eq = [[1, 1, 1, 0]]
    b_eq = [k]

    bounds = [(m, None)] * 3 + [(None, None)]

    res = linprog(c=c_obj, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    return res.x[:-1]

def minimaxfit(x,y):
    """
    :param x: [[x1], [x_2], ..., [xn]]
    :param y: [[y1], [y2], ..., [yn]]
    :return: a, b, r
    """

    n, m = x.shape

    c = np.zeros(n + 2)
    c[-1] = 1
    bounds = [(None, None) for _ in range(n + 2)]

    A_ub = []
    b_ub = []

    for i in range(m):
        xi = x[:, i]
        yi = y[0][i]

        # + (a*xi + b - yi) <= t  =>  a*xi + b - t <= yi
        A_ub.append(np.append(xi, [1, -1]))
        b_ub.append(yi)

        # - (a*xi + b - yi) <= t  =>  -a*xi - b - t <= -yi
        A_ub.append(np.append(-xi, [-1, -1]))
        b_ub.append(-yi)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    a = res.x[:n]
    b = res.x[n]
    r = res.x[n + 1]
    return a, b, r


if __name__ == '__main__':
    pass
