import numpy as np


class FcLayer(object):
    def __init__(self, network, node_count, activator, momentum_rate=0.0):
        self.output_array = np.zeros([node_count])
        self.bias_array = np.zeros([node_count])
        self.bias_gradient_cache = np.zeros([node_count])
        input_shape = network.layers[-1].get_output().shape
        self.input_1dim = reduce(lambda ret, dim: ret * dim, input_shape, 1)
        self.trans_matrix = np.random.uniform(-1e-4, 1e-4, [node_count, self.input_1dim])
        self.trans_gradient_cache = np.zeros(self.trans_matrix.shape)
        self.activator = activator
        self.momentum_rate = momentum_rate
        network.append_layer(self)

    def get_output(self):
        return self.output_array

    def forward(self, input_array, training=False):
        self.input_array = input_array.reshape(self.input_1dim)
        self.output_array = np.dot(self.trans_matrix, self.input_array) + self.bias_array
        self.output_array = np.array([self.activator.forward(value) for value in self.output_array])

    def calc_output_layer_delta(self, label):
        derivative = np.array([self.activator.backward(out) for out in self.output_array])
        self.delta_array = derivative * (label - self.output_array)

    def calc_layer_delta(self, downstream_layer):
        derivative = np.array([self.activator.backward(out) for out in self.output_array])
        self.delta_array = derivative * downstream_layer.get_transformed_delta()

    def update_weight(self, rate):
        trans_gradient = np.dot(self.delta_array.reshape([len(self.delta_array),1]),self.input_array.reshape([1, self.input_1dim]))
        bias_gradient = self.delta_array
        self.trans_matrix += rate * trans_gradient + self.momentum_rate * self.trans_gradient_cache
        self.bias_array += rate * bias_gradient + self.momentum_rate * self.bias_gradient_cache
        self.trans_gradient_cache = trans_gradient
        self.bias_gradient_cache = bias_gradient

    def get_transformed_delta(self):
        return np.dot(self.trans_matrix.transpose(), self.delta_array)






