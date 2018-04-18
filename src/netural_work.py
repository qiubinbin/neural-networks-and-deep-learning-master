"""源程序，二次平方代价函数"""
import numpy as np
import random


def sigmoid(z):
    """激励函数"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """sigmoid(z)函数的导数"""
    return sigmoid(z) * (1 - sigmoid(z))


class Network():
    def __init__(self, sizes):
        """初始化网络结构"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """向前传播"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        # print('result:', np.sum(a))
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """随机梯度下降算法"""
        """epochs表示迭代期数；training_data为元组列表，表示训练输入和预期输出；
        mini_batch_size表示采样时的小批量数据的大小；eta为学习效率η"""
        test_data = list(test_data)
        training_data = list(training_data)
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):  # 迭代期
            random.shuffle(training_data)  # 打乱顺序
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.updata_mini_batch(mini_batch, eta)
            if test_data:
                print('Epoch: {0} : {1} / {2}'.format(j, self.evaluate(test_data), n_test))
            else:
                print('Epoch {0} complete'.format(j))

    def updata_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_babla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_babla_w)]
            self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]  # 存储每一层的激活向量,activations[-1]为输出
        zs = []  # 存储Z向量,对应每对数据，一层一层的向量
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_result = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_result)

    def cost_derivative(self, output_activations, y):
        """返回预期输出和实际输出之差"""
        return (output_activations - y)
