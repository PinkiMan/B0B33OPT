import numpy as np
import matplotlib.pyplot as plt





def fit_wages(t,M):
    A = np.vstack([np.ones(len(t)), t]).T
    x = np.linalg.lstsq(A,M)[0]
    return x

def quarter2_2009(x):
    return x[0] + x[1]*2009.25

with open('du1-mzda/mzda/mzdy.txt', 'r') as FILE:
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

x = fit_wages(t,M)
m,c = x

y = quarter2_2009(x)

plt.plot(t, c*t + m, 'r', label='Fitted line')
plt.scatter(2009.25, y)
plt.legend()
plt.show()


