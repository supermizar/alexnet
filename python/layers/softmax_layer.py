import numpy as np

from fc_layer import FcLayer
from python.utils.activator import Softmax


class SoftmaxLayer(FcLayer):

    def __init__(self, network, node_count):
        FcLayer.__init__(self, network, node_count, Softmax())

    def forward(self, input_array, training=False):
        self.input_array = input_array.reshape(self.input_1dim)
        self.output_array = np.dot(self.trans_matrix, self.input_array) + self.bias_array
        self.output_array = self.activator.forward(self.output_array)

    def calc_output_layer_delta(self, label):
        self.delta_array = self.output_array - label

    def update_weight(self, rate):
        self.trans_matrix -= rate * np.dot(self.delta_array.reshape([len(self.delta_array),1]),self.input_array.reshape([1, self.input_1dim]))
        self.bias_array -= rate * self.delta_array