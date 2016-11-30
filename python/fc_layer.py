import numpy as np


class FcLayer(object):
    def __init__(self, upstream_output_shape, node_count):
        self.output_array = np.zeros([node_count])
        self.input_shape = upstream_output_shape
        self.input_1dim = reduce(lambda ret, dim: ret * dim, upstream_output_shape, 1)
        self.trans_matrix = np.zeros([node_count, self.input_1dim])

    def get_output(self):
        return self.output_array

    def forward(self, input_array):
        input_array = input_array.reshape(self.input_1dim)
        self.output_array = np.dot(self.trans_matrix, input_array)





