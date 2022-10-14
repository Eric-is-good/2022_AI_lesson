import numpy as np


def data_loader(addr):
    datas = []

    with open(addr) as f:
        for line in f.readlines():
            line = line.split()
            line_num = []
            for i in line:
                line_num.append(float(i))
            datas.append(line_num)
    return datas


if __name__ == '__main__':
    datas = data_loader("../../data/traindata.txt")
    print(datas)

    datas = data_loader("../../data/testdata.txt")
    print(datas)

