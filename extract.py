import numpy as np

district_map = {}
with open('cluster_map/cluster_map') as f:
    for line in f:
        p = line.strip().split('\t')
        district_map[p[0]] = int(p[1]) - 1
assert(len(district_map) == 66)

def time_convert_to_id(time):
    a = time.strip().split()[1].split(':')
    return int(a[0]) * 6 + int(a[1]) / 10

def parse_order(fname):
    with open(fname) as f:
        for line in f:
            p = line.strip().split('\t')
            yield (p[1] != 'NULL', district_map[p[3]], time_convert_to_id(p[6]))

def extract_order(dates, pattern):
    for i in dates:
        x = np.zeros((66, 144)) 
        y = np.zeros((66, 144)) 
        a = parse_order(pattern % i)
        for aa in a:
            if aa[0] == True: x[aa[1]][aa[2]] += 1
            else: y[aa[1]][aa[2]] += 1
        for j in range(66):
            for k in range(144):
                print("%s\t%s\t%s\t%s\t%s" % (i, j + 1, k + 1, x[j][k], y[j][k]))

def parse_traffic(fname):
    with open(fname) as f:
        for line in f:
            p = line.strip().split('\t')
            d = district_map[p[0]]
            t = time_convert_to_id(p[-1])
            r = []
            for i in range(1, 5):
                r.append(int(p[i].split(':')[1]))
            yield (d, t) + tuple(r)

def extract_traffic(dates, pattern):
    for i in dates:
        x = [[('0', '0', '0', '0') for i1 in range(144)] for i2 in range(66)]
        a = parse_traffic(pattern % i)
        for aa in a:
            x[aa[0]][aa[1]] = (str(aa[2]), str(aa[3]), str(aa[4]), str(aa[5]))
        for j in range(66):
            for k in range(144):
                print("%s\t%s\t%s\t%s" % (i, j + 1, k + 1, '\t'.join(x[j][k])))

def parse_weather(fname):
    with open(fname) as f:
        for line in f:
            p = line.strip().split('\t')
            yield (time_convert_to_id(p[0]), int(p[1]), float(p[2]), float(p[3]))

def extract_weather(dates, pattern):
    for i in dates:
        a = parse_weather(pattern % i)
        for aa in a:
            print("%s\t%s\t%s\t%s\t%s" % (i, aa[0] + 1, aa[1], aa[2], aa[3]))


if __name__ == '__main__':
    #extract_weather(range(1, 22), './weather_data/weather_data_2016-01-%02d')

    #extract_traffic(range(1, 22), './traffic_data/traffic_data_2016-01-%02d')
    #extract_traffic(range(23, 32, 2), '../../season_2/test_set_2/traffic_data/traffic_data_2016-01-%02d_test')

    #extract_order(range(1, 22), './order_data/order_data_2016-01-%02d')
    #extract_order(range(22, 31, 2), '../test_set_1/order_data/order_data_2016-01-%02d_test')
    extract_order(range(23, 32, 2), '../../season_2/test_set_2/order_data/order_data_2016-01-%02d_test')



