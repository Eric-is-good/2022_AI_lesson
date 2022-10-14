from sklearn import tree
from io import StringIO
import pydotplus
import numpy as np
from dataloader import data_loader

traindata = data_loader("../../data/traindata.txt")
testdata = data_loader("../../data/testdata.txt")

tree_model = tree.DecisionTreeClassifier(criterion='gini',
                                         max_depth=None,
                                         min_samples_leaf=1,
                                         ccp_alpha=0)
traindata = np.array(traindata)
train_X = traindata[:, :-1]
train_Y = traindata[:, -1:]

testdata = np.array(testdata)
test_X = testdata[:, :-1]
test_Y = testdata[:, -1:]

tree_model.fit(train_X, train_Y)
result = tree_model.score(test_X, test_Y)
print(result)

dot_data = StringIO()
feature_names = ["a", "b", "c", "d"]
target_names = ["1", "2", "3"]
tree.export_graphviz(tree_model,
                     out_file=dot_data,
                     feature_names=feature_names,
                     class_names=target_names,
                     filled=True,
                     rounded=True,
                     special_characters=True)

print(dot_data.getvalue())

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("../../out/default.pdf")
