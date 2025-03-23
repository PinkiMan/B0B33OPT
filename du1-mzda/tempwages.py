import numpy as np

def fit_wages(t,M):
    A = np.vstack([np.ones(len(t)), t]).T
    x = np.linalg.lstsq(A,M)[0]
    return x

def quarter2_2009(x):
    return x[0] + x[1]*2009.25

def fit_temps(t, T, omega):
    A = np.vstack([np.ones(len(t)), t, np.sin(omega * t), np.cos(omega * t)]).T
    x = np.linalg.lstsq(A, T)[0]

    return x