# 这是我在电子科技大学上的人工智能课





## 实验一        决策树



挺简单的，就一个递归即可

![img](https://github.com/Eric-is-good/2022_AI_lesson/blob/main/%E5%AE%9E%E9%AA%8C%E4%B8%80%EF%BC%88%E5%86%B3%E7%AD%96%E6%A0%91%EF%BC%89/img.png)

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





## 实验二   BP 算法

按照 pytorch 风格使用 numpy 重写前向后向

```
class Linear(BaseNetwork):
    def __init__(self, inplanes, outplanes):
        super(Linear, self).__init__()
        self.weight = np.random.rand(inplanes, outplanes) * 2 - 1
        self.bias = np.random.rand(outplanes) * 2 - 1
        self.input = None
        self.output = None
        self.wgrad = np.zeros(self.weight.shape)
        self.bgrad = np.zeros(self.bias.shape)
        self.variable = Variable(self.weight, self.wgrad, self.bias, self.bgrad)

    def parameters(self):
        return self.variable

    def forward(self, *x):
        x = x[0]
        self.input = x
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output

    def backward(self, grad):
        self.bgrad = grad
        self.wgrad += np.dot(self.input.T, grad)
        grad = np.dot(grad, self.weight.T)
        return grad
```

训练也模仿 pytorch

```
optimizer.zero_grad()
pred = net(x)
loss = criterion(pred, y)
net.backward()
optimizer.step()
```