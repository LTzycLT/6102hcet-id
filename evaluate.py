import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn import svm
#from plot_learning_curve import plot_learning_curve

def is_weekend(date):
    if (date - 2) % 7 == 0: return 1
    if (date - 3) % 7 == 0: return 1
    return 0

amount = [ 24720,   7960,   8789,  10538,   7453,   4834,   4213,  11471,   7562, 10420,  20138,   9674,  12635,  11625,  20005,  17148,   8946,  14422, 23817,  29968,  17080,  25528,  15513,  10209,  25207,  26259,  16557, 24230,  35096,  18524,  23213]
amount = np.array(amount)

a = pd.read_table('./intermediate_data/order', names = ['date', 'district', 'time', 'available', 'gap']).astype(int).values
A = np.zeros((2, 66, 144))
B = np.zeros((2, 66, 144))
for i in range(len(a)):
    if a[i][0] == 1: continue
    j = is_weekend(a[i][0])
    k = a[i][1] - 1
    l = a[i][2] - 1
    A[j][k][l] += a[i][-1]
    B[j][k][l] += 1 
A /= B

def prepare_data(suffix=""):
    a = pd.read_table('./intermediate_data/order%s' % suffix, names = ['date', 'district', 'time', 'available', 'gap'])
    a = a.astype(int).values

    b = pd.read_table('./intermediate_data/traffic%s' % suffix, names = ['date', 'district', 'time', 't0', 't1', 't2', 't3'])
    b = b.astype(int).values

    X = []
    y = []
    d = []
    for i in range(0, len(a)):
        if a[i][0] == 1 or a[i - 3][2] >= a[i][2]: continue
        j = is_weekend(a[i][0])
        k = a[i][1] - 1
        l = a[i][2] - 1
        X.append([amount[a[i][0] - 1], A[j][k][l], a[i - 1][-1], a[i - 2][-1], a[i - 3][-1], b[i][-2], b[i][-1]])
        y.append(a[i][-1])
        d.append((a[i][0], a[i][1], a[i][2]))
    return np.array(X), np.array(y), d
    

def output(X, y, d):
    zyc = {}
    for i in range(X.shape[0]):
        if d[i][0] not in zyc: zyc[d[i][0]] = np.zeros((66, 144))
        d_id = d[i][1] - 1
        t_id = d[i][2] - 1
        zyc[d[i][0]][d_id][t_id] = y[i]

    lines = open('../../season_2/test_set_2/read_me_2.txt').read().splitlines()
    for i in range(1, len(lines)):
        line = lines[i]
        p = line.strip().split('-')
        date = int(p[2])
        t_id = int(p[3]) - 1
        for j in range(66):
            x = zyc[date][j][t_id]
            print("%s,%s,%s" % (j + 1, line.strip(), x))

def scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return np.abs(y_pred - y).sum() / y.shape[0]

if __name__ == '__main__':

    X_train, y_train, d = prepare_data()
    weight = []
    for yy in y_train: weight.append(0 if yy == 0 else 100 / yy)

    model = ensemble.GradientBoostingRegressor()
    model.fit(X_train, y_train, sample_weight=weight)

    #y_pred = model.predict(X_train) 
    #y_pred[y_pred < 1] = 1
    #for i in range(len(X_train)):
    #    print("%s\t%s\t%s\t%s\t%s" % (d[i][0], d[i][1], d[i][2], y_train[i], y_pred[i]))

    X, y, d = prepare_data("_test_2")
    y_pred = model.predict(X) 
    y_pred[y_pred < 1] = 1
    #for i in range(len(X)):
    #    print("%s\t%s\t%s\t%s\t%s" % (d[i][0], d[i][1], d[i][2], y[i], y_pred[i]))
    output(X, y_pred, d)
