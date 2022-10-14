import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from treePlotter import ID3_Tree


class DecisionTree:
    def __init__(self,
                 train_addr="../../data/traindata.txt",
                 test_addr="../../data/testdata.txt",
                 continuous_features=[],
                 ):
        self.info = "default is i3d, default all the features is not continuous"

        self.Tree = None

        self.train_addr = train_addr
        self.test_addr = test_addr
        self.continuous_features = continuous_features

        self.dataloader()

    def dataloader(self):
        self.traindata = pd.read_csv(self.train_addr)
        self.testdata = pd.read_csv(self.test_addr)

        self.column_count = dict(
                [(ds, list(pd.unique(self.traindata[ds]))) for ds in self.traindata.iloc[:, :-1].columns])

        return self.traindata, self.testdata, self.column_count

    ##################################################################################
    #############################   calculation part   ###############################

    # calculate ent
    def cal_information_entropy(self, data):
        """
        Ent(D) = -∑ Pk*log2(Pk) k=1..len(label_set)
        :param label_list:
        :return:
        """
        data_label = data.iloc[:, -1]
        label_class = data_label.value_counts()
        Ent = 0
        for k in label_class.keys():
            p_k = label_class[k] / len(data_label)
            Ent += -p_k * np.log2(p_k)
        return Ent

    # calculate ent gain for given feature when data is dispersed
    def cal_information_gain(self, data, feature):
        """
        gain = - (Ent_d - Ent)
        :param data:
        :param feature:
        :return:
        """
        Ent = self.cal_information_entropy(data)

        # Count and sort each value in Series, return [(key_name, num_of_key_appear),....]
        feature_class = data[feature].value_counts()

        Ent_d = 0
        for v in feature_class.keys():
            weight = feature_class[v] / data.shape[0]
            Ent_v = self.cal_information_entropy(data.loc[data[feature] == v])
            Ent_d += weight * Ent_v
        return Ent - Ent_d

    # calculate ent gain for given feature when data is continuous (default)
    def cal_information_gain_continuous(self, data, feature):
        data_a_value = data[feature].values
        data_a_value = sorted(list(set(data_a_value)))
        n = len(data_a_value)

        Ent = self.cal_information_entropy(data)  # old Ent(D)

        select_points = []
        for i in range(n - 1):
            val = (data_a_value[i] + data_a_value[i + 1]) / 2  # get the middle
            data_left = data.loc[data[feature] < val]
            data_right = data.loc[data[feature] > val]
            ent_left = self.cal_information_entropy(data_left)
            ent_right = self.cal_information_entropy(data_right)
            result = Ent - len(data_left) / len(data) * ent_left - len(data_right) / len(data) * ent_right
            select_points.append([val, result])
        select_points.sort(key=lambda x: x[1], reverse=True)  # sort by the gain
        return select_points[0][0], select_points[0][1]  # return  max gain : (point, gain)

    def get_most_label(self, data):
        data_label = data.iloc[:, -1]
        label_sort = data_label.value_counts(sort=True)
        return label_sort.keys()[0]

    def get_best_feature(self, data):
        features = data.columns[:-1]
        res = {}
        for a in features:
            if a in self.continuous_features:
                temp_val, temp = self.cal_information_gain_continuous(data, a)
                res[a] = [temp_val, temp]
            else:
                temp = self.cal_information_gain(data, a)
                res[a] = [-1, temp]  # The discrete value has no dividing point ,which is replaced by - 1

        res = sorted(res.items(), key=lambda x: x[1][1], reverse=True)
        return res[0][0], res[0][1][0]

    def Encode_discrete_data(self, data, best_feature):
        attr = pd.unique(data[best_feature])
        new_data = [(nd, data[data[best_feature] == nd]) for nd in attr]
        new_data = [(n[0], n[1].drop([best_feature], axis=1)) for n in new_data]
        return new_data

    ##################################################################################
    #############################   train and predict   ##############################

    def train(self, data=None):
        if data is None:
            data = self.traindata

        data_label = data.iloc[:, -1]

        # There is only one type
        if len(data_label.value_counts()) == 1:
            return data_label.values[0]

        # The characteristic values of all data are the same,
        # and the class with the most samples is selected as the classification result
        if all(len(data[i].value_counts()) == 1 for i in data.iloc[:, :-1].columns):
            return self.get_most_label(data)

        best_feature, best_feature_val = self.get_best_feature(data)  # use best gain to get the best feature

        if best_feature in self.continuous_features:
            node_name = best_feature + '<' + str(best_feature_val)
            Tree = {node_name: {}}
            Tree[node_name]['是'] = self.train(data.loc[data[best_feature] < best_feature_val])
            Tree[node_name]['否'] = self.train(data.loc[data[best_feature] > best_feature_val])
        else:
            Tree = {best_feature: {}}
            exist_vals = pd.unique(data[best_feature])  # 当前数据下最佳特征的取值
            if len(exist_vals) != len(self.column_count[best_feature]):  # 如果特征的取值相比于原来的少了
                no_exist_attr = set(self.column_count[best_feature]) - set(exist_vals)  # 少的那些特征
                for no_feat in no_exist_attr:
                    Tree[best_feature][no_feat] = self.get_most_label(data)  # 缺失的特征分类为当前类别最多的
            for item in self.Encode_discrete_data(data, best_feature):  # 根据特征值的不同递归创建决策树
                Tree[best_feature][item[0]] = self.train(item[1])

        self.Tree = Tree
        return Tree

    def excel_to_dictlist(self, data):
        dictlist = []
        for i in range(data.shape[0]):
            dictlist.append(data.loc[i].to_dict())
        return dictlist

    def predict(self, data):
        result = []
        dictlist = self.excel_to_dictlist(data)
        for item in dictlist:
            result.append(self.predict_one(item))
        return result

    def predict_one(self, each_test_data, left_tree=None):
        if left_tree is None:
            left_tree = self.Tree

        first_feature = list(left_tree.keys())[0]
        feature_name = first_feature.split('<')[0]
        if feature_name in self.continuous_features:
            second_dict = left_tree[first_feature]
            val = float(first_feature.split('<')[-1])
            input_first = each_test_data.get(feature_name)
            if input_first < val:
                left_tree = second_dict['是']
            else:
                left_tree = second_dict['否']
        else:
            second_dict = left_tree[first_feature]
            input_first = each_test_data.get(first_feature)
            left_tree = second_dict[input_first]

        if isinstance(left_tree, dict):
            class_label = self.predict_one(each_test_data, left_tree)
        else:
            class_label = left_tree
        return class_label

    def score(self, data=None):
        if data is None:
            data = self.testdata
        result = self.predict(data)
        result = np.array(result)
        ground_trues = data.iloc[:, -1:]
        ground_trues = np.array(ground_trues).ravel()
        score = np.sum(result == ground_trues) / result.shape[0]
        return score

    def plot_tree(self):
        ID3_Tree(self.Tree)


if __name__ == '__main__':
    tree = DecisionTree(
        train_addr="../../data/data_word.csv",
        test_addr="../../data/data_word.csv",
        # continuous_features=["1", "2", "3", "4", "5"]
    )
    tree.train()
    print(tree.Tree)
    print(tree.score())
    tree.plot_tree()
