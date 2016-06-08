import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble

#dates = {}
#with open('./intermediate_data/order') as f:
#    for line in f:
#        p = [int(float(a)) for a in line.strip().split('\t')]
#        if p[0] not in dates: dates[p[0]] = np.zeros((66, 144))
#        dates[p[0]][p[1] - 1][p[2] - 1] = p[4]
#
#def predict(date, d_id, t_id):
#    while date not in dates: date -= 7
#    v = dates[date][d_id][t_id]
#    return max(0, v - 2)
#
#    #u = float(p[4])
#    #c1 = 0
#    #c2 = 0
#    #while pre_date > 0 and c1 < 1:
#    #    if pre_date in dates: 
#    #        v += dates[pre_date][d_id][t_id]
#    #        c1 += 1
#    #    pre_date -= 7
#    #pre_date = p[0]
#    #while pre_date > 0 and c2 < 0:
#    #    if pre_date in dates: 
#    #        v += dates[pre_date][d_id][t_id]
#    #        c2 += 1
#    #    pre_date -= 1 
#    #v /= (c1 + c2)
#    #res[d_id].append(abs(u - v) / u)
#

def is_weekend(date):
    if (date - 2) % 7 == 0: return 1
    if (date - 3) % 7 == 0: return 1
    return 0

def prepare_data(suffix=""):
    a = pd.read_table('./intermediate_data/order%s' % suffix, names = ['date', 'district', 'time', 'available', 'gap'])
    a = a.astype(int).values

    b = pd.read_table('./intermediate_data/traffic%s' % suffix, names = ['date', 'district', 'time', 't0', 't1', 't2', 't3'])
    b = b.astype(int).values

    X_all = []
    X_available = []
    d = []
    y_all = []
    y_available = []
    for i in range(0, len(a)):
        if a[i][0] == 1 or a[i - 3][2] >= a[i][2]: continue
        X_all.append([is_weekend(a[i][0]), a[i][1], a[i][2], a[i - 1][-1] + a[i - 1][-2], a[i - 2][-1] + a[i - 2][-2], a[i - 3][-1] + a[i - 3][-2], b[i][-4], b[i][-3], b[i][-2], b[i][-1]])
        X_available.append([is_weekend(a[i][0]), a[i][1], a[i][2], a[i - 1][-2], a[i - 2][-2], a[i - 3][-2], b[i][-4], b[i][-3], b[i][-2], b[i][-1]])
        y_all.append(a[i][-1] + a[i][-2])
        y_available.append(a[i][-2])
        d.append((a[i][0], a[i][1], a[i][2]))
    return np.array(X_all), np.array(X_available), np.array(y_all), np.array(y_available), d
    

def output(X, y, d):
    zyc = {}
    for i in range(X.shape[0]):
        if d[i][0] not in zyc: zyc[d[i][0]] = np.zeros((66, 144))
        d_id = d[i][1] - 1
        t_id = d[i][2] - 1
        zyc[d[i][0]][d_id][t_id] = y[i]

    lines = open('../test_set_1/read_me_1.txt').read().splitlines()
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

    X_all_train, X_available_train, y_all_train, y_available_train, d_train = prepare_data()

    m1 = ensemble.GradientBoostingRegressor()
    m2 = ensemble.GradientBoostingRegressor()

    m1.fit(X_all_train, y_all_train)
    m2.fit(X_available_train, y_available_train)

    y_all_pred = m1.predict(X_all_train)
    y_available_pred = m2.predict(X_available_train)

    y_pred = np.maximum(0, y_all_pred - y_available_pred)
    y_real = np.maximum(0, y_all_train - y_available_train)
    print(np.abs(y_pred - y_real).sum() / y_pred.shape[0])




    #scores = cross_validation.cross_val_score(model, X_train, y_train, cv=5, scoring = scorer)
    #print(scores)


    #X, d = prepare_test()
    #y = model.predict(X) 
    #y[y < 1] = 1
    ##for i in range(len(X)):
    ##    print("%s\t%s\t%s\t%s" % (d[i][0], d[i][1], d[i][2], y[i]))
    #output(X, y, d)
