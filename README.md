# 这是我在电子科技大学上的人工智能课





## 实验一        决策树



挺简单的，就一个递归即可

![img](C:\Users\Administrator\Desktop\python\2022_AI_lesson\实验一（决策树）\img.png)

提供以下 API

```
tree = DecisionTree(
    train_addr="../../data/data_word.csv",
    test_addr="../../data/data_word.csv",
    # continuous_features=["1", "2", "3", "4", "5"]
)

tree.train()

tree.predict()

score = tree.score()

tree.plot_tree()
```