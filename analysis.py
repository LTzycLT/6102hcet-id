import numpy as np
import matplotlib.pyplot as plt

A = np.zeros((31, 66, 144))
G = np.zeros((31, 66, 144))
T = np.zeros((31, 66, 144))
W = np.zeros((31, 144))

segments_raw = [58, 70, 82, 94, 106, 118, 130, 142]
segments = []
for x in segments_raw:
    for i in range(4, 1, -1): segments.append(x - i)
segments = np.array(segments)

with open("./intermediate_data/order") as f:
    for line in f:
        p = line.strip().split()
        p[0] = int(p[0]) - 1
        p[1] = int(p[1]) - 1
        p[2] = int(p[2]) - 1
        A[p[0]][p[1]][p[2]] = float(p[3])
        G[p[0]][p[1]][p[2]] = float(p[4])
with open("./intermediate_data/order_test_1") as f:
    for line in f:
        p = line.strip().split()
        p[0] = int(p[0]) - 1
        p[1] = int(p[1]) - 1
        p[2] = int(p[2]) - 1
        A[p[0]][p[1]][p[2]] = float(p[3])
        G[p[0]][p[1]][p[2]] = float(p[4])
with open("./intermediate_data/order_test_2") as f:
    for line in f:
        p = line.strip().split()
        p[0] = int(p[0]) - 1
        p[1] = int(p[1]) - 1
        p[2] = int(p[2]) - 1
        A[p[0]][p[1]][p[2]] = float(p[3])
        G[p[0]][p[1]][p[2]] = float(p[4])
print(G[:, :, segments].sum(axis = (1, 2)))

#G = G.sum(axis=1)[:, segments]
#plt.plot(np.arange(21 * 144), R[:, district, :].reshape(-1))

plt.plot(np.arange(G.shape[0]), G.sum(axis = (1, 2)))
#plt.show()

#plt.xlabel('time (s)')
#plt.ylabel('voltage (mV)')
#plt.title('About as simple as it gets, folks')
#plt.grid(True)
#plt.savefig("test.png")
