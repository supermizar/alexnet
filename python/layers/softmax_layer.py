import numpy as np

from fc_layer import FcLayer
from python.utils.activator import Softmax


class SoftmaxLayer(FcLayer):

    def __init__(self, network, node_count, momentum_rate=0.0):
        FcLayer.__init__(self, network, node_count, Softmax(), momentum_rate=momentum_rate)

    def forward(self, input_array, training=False):
        self.input_array = input_array.reshape(self.input_1dim)
        self.output_array = np.dot(self.trans_matrix, self.input_array) + self.bias_array
        self.output_array = self.activator.forward(self.output_array)

    def calc_output_layer_delta(self, label):
        self.delta_array = label - self.output_array

    def update_weight(self, rate):
        trans_gradient = np.dot(self.delta_array.reshape([len(self.delta_array),1]),self.input_array.reshape([1, self.input_1dim]))
        bias_gradient = self.delta_array
        self.trans_matrix += rate * trans_gradient + self.momentum_rate * self.trans_gradient_cache
        self.bias_array += rate * bias_gradient + self.momentum_rate * self.bias_gradient_cache
        self.trans_gradient_cache = trans_gradient
        self.bias_gradient_cache = bias_gradient
