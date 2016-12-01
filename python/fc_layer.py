import numpy as np


class FcLayer(object):
    def __init__(self, network, node_count, activator):
        self.output_array = np.zeros([node_count])
        self.input_shape = network.layers[-1].get_output().shape
        self.input_1dim = reduce(lambda ret, dim: ret * dim, self.input_shape, 1)
        self.trans_matrix = np.zeros([node_count, self.input_1dim])
        self.activator = activator
        network.append_layer(self)

    def get_output(self):
        return self.output_array

    def forward(self, input_array):
        self.input_array = input_array.reshape(self.input_1dim)
        self.output_array = np.dot(self.trans_matrix, self.input_array)
        self.output_array = np.array([self.activator.forward(value) for value in self.output_array])

    def calc_output_layer_delta(self, label):
        derivative = np.array([self.activator.backward(out) for out in self.output_array])
        self.delta_array = derivative * (label - self.output_array)

    def calc_hidden_layer_delta(self, downstream_layer):
        derivative = np.array([self.activator.backward(out) for out in self.output_array])
        self.delta_array = derivative * downstream_layer.get_transformed_delta()

    def update_weight(self, rate):
        self.trans_matrix += rate * np.dot(self.delta_array.reshape([len(self.delta_array),1]),self.input_array.reshape([1, self.input_1dim]))

    def get_transformed_delta(self):
        return np.dot(self.trans_matrix.transpose(), self.delta_array)






