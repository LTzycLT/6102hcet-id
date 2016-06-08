import pandas as pd
import numpy as np
A = np.zeros((21, 66, 144))
a = pd.read_table('./intermediate_data/order', names = ['date', 'district', 'time', 'available', 'gap']).astype(int).values
for i in range(len(a)): A[a[i][0] - 1][a[i][1] - 1][a[i][2] - 2] = a[i][-1]

lines = open('../test_set_1/read_me_1.txt').read().splitlines()
for i in range(1, len(lines)):
    line = lines[i]
    p = line.strip().split('-')
    date = int(p[2])
    t_id = int(p[3]) - 1

    for j in range(66):
        x = [str(A[date][j][t_id]) for date in range(1, 21)]
        print("%s,%s,%s" % (j + 1, line.strip(), '\t'.join(x)))


