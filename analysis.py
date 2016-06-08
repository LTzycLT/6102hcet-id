import numpy as np
import matplotlib.pyplot as plt

O = np.zeros((21, 66, 144))
A = np.zeros((21, 66, 144))
T = np.zeros((21, 66, 144))
W = np.zeros((21, 144))

with open("./intermediate_data/order") as f:
    for line in f:
        p = line.strip().split()
        p[0] = int(p[0]) - 1
        p[1] = int(p[1]) - 1
        p[2] = int(p[2]) - 1
        O[p[0]][p[1]][p[2]] = float(p[3]) + float(p[4])
        A[p[0]][p[1]][p[2]] = float(p[4])
#with open("./intermediate_data/traffic") as f:
#    for line in f:
#        p = line.strip().split()
#        p[0] = int(p[0]) - 1
#        p[1] = int(p[1]) - 1
#        p[2] = int(p[2]) - 1
#        T[p[0]][p[1]][p[2]] = float(p[3])
#with open("./intermediate_data/weather") as f:
#    for line in f:
#        p = line.strip().split()
#        p[0] = int(p[0]) - 1
#        p[1] = int(p[1]) - 1
#        W[p[0]][p[1]] = float(p[4])




#plt.plot(np.arange(21 * 144), R[:, district, :].reshape(-1))

#plt.plot(np.arange(66), A.sum(axis = (0, 2)))
#plt.plot(np.arange(66), O.sum(axis = (0, 2)))

a = A[:, 50, 100]
plt.plot(np.arange(a.shape[0]), a)
plt.show()

#plt.xlabel('time (s)')
#plt.ylabel('voltage (mV)')
#plt.title('About as simple as it gets, folks')
#plt.grid(True)
#plt.savefig("test.png")
