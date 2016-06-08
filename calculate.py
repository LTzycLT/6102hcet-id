import numpy as np
import pandas as pd

R = np.zeros((21, 66, 144))
P = np.zeros((21, 66, 144))
a = np.array(pd.read_table('./out', names = ['date', 'district', 'time', 'real', 'pred']).astype(int).values)
assert(a.shape[0] == 66 * 20 * 141)
for i in range(a.shape[0]):
    j = a[i][0] - 1;
    k = a[i][1] - 1;
    l = a[i][2] - 1;
    R[j][k][l] = a[i][-2]
    P[j][k][l] = a[i][-1]


a = np.zeros(66)
for i in range(21):
    for j in range(66):
        for k in range(144):
            if R[i][j][k] > 0:
                a[j] += abs(R[i][j][k] - P[i][j][k]) / R[i][j][k] 

print((a / (20 * 141)).sum() / 66)


