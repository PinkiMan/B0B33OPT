import numpy as np
import matplotlib.pyplot as plt


def fit_temps(t, T, omega):
    A = np.vstack([np.ones(len(t)), t, np.sin(omega * t), np.cos(omega * t)]).T
    x = np.linalg.lstsq(A, T)[0]

    return x



"""with open('teplota.txt', 'r') as FILE:
    t = []
    M = []

    lines = FILE.readlines()

    for line in lines:
        data = line.strip().split(' ')
        t.append(float(data[0]))
        M.append(float(data[1]))

    t = np.array(t)
    M = np.array(M)

plt.plot(t, M, 'o', label='Original data', markersize=1)

omega = (2 * np.pi)/(365)
x = fit_temps(t,M, omega)
x1, x2, x3, x4 = x


plt.plot(t, x1 + x2*t + x3*np.sin(omega*t) + x4*np.cos(omega*t), 'r', label='Fitted line')
plt.legend()
plt.show()"""
