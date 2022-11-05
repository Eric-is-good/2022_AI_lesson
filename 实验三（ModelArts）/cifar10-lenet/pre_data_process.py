import numpy as np
import pickle

from matplotlib import pyplot as plt


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



if __name__ == '__main__':

    # # pickle 处理部分
    #
    # labellist = []
    # datalist = []
    # data1 = unpickle("./cifar-10-batches-py/1")
    # for d in data1[b'data']:
    #     datalist.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))
    # for l in data1[b'labels']:
    #     hot = np.zeros(10)
    #     hot[int(l)] = 1
    #     labellist.append(hot)
    #
    # data1 = unpickle("./cifar-10-batches-py/data_batch_2")
    # for d in data1[b'data']:
    #     datalist.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))
    # for l in data1[b'labels']:
    #     hot = np.zeros(10)
    #     hot[int(l)] = 1
    #     labellist.append(hot)
    #
    # data1 = unpickle("./cifar-10-batches-py/data_batch_3")
    # for d in data1[b'data']:
    #     datalist.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))
    # for l in data1[b'labels']:
    #     hot = np.zeros(10)
    #     hot[int(l)] = 1
    #     labellist.append(hot)
    #
    # data1 = unpickle("./cifar-10-batches-py/data_batch_4")
    # for d in data1[b'data']:
    #     datalist.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))
    # for l in data1[b'labels']:
    #     hot = np.zeros(10)
    #     hot[int(l)] = 1
    #     labellist.append(hot)
    #
    # data1 = unpickle("./cifar-10-batches-py/data_batch_5")
    # for d in data1[b'data']:
    #     datalist.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))
    # for l in data1[b'labels']:
    #     hot = np.zeros(10)
    #     hot[int(l)] = 1
    #     labellist.append(hot)
    #
    # labels = np.asarray(labellist)
    # datas = np.asarray(datalist)
    #
    # traindata = dict()
    # traindata[b'data'] = datas
    # traindata[b'labels'] = labels
    #
    # with open("train_batch", "wb") as f:
    #     pickle.dump(traindata, f)
    #
    #
    #
    #
    # labellist = []
    # datalist = []
    # data1 = unpickle("./cifar-10-batches-py/test_batch")
    # for d in data1[b'data']:
    #     datalist.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))
    # for l in data1[b'labels']:
    #     hot = np.zeros(10)
    #     hot[int(l)] = 1
    #     labellist.append(hot)
    #
    # print(len(labellist))
    #
    # labels = np.asarray(labellist)
    # datas = np.asarray(datalist)
    #
    # traindata = dict()
    # traindata[b'data'] = datas
    # traindata[b'labels'] = labels
    #
    #
    # with open("test_batch", "wb") as f:
    #     pickle.dump(traindata, f)



    # 生成一个小批量
    labellist = []
    datalist = []
    data1 = unpickle("./1")
    for d in data1[b'data']:
        datalist.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))
    for l in data1[b'labels']:
        hot = np.zeros(10)
        hot[int(l)] = 1
        labellist.append(hot)

    print(len(labellist))

    labels = np.asarray(labellist[:100])
    datas = np.asarray(datalist[:100])

    traindata = dict()
    traindata[b'data'] = datas
    traindata[b'labels'] = labels


    with open("mini_batch", "wb") as f:
        pickle.dump(traindata, f)





