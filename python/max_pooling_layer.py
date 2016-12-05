import numpy as np
from utils import get_patch
from utils import get_max_index


class MaxPoolingLayer(object):
    def __init__(self, network,filter_width,
                 filter_height, stride):
        upstream_layer = network.layers[-1]
        self.input_width = upstream_layer.output_width
        self.input_height = upstream_layer.output_height
        self.channel_number = upstream_layer.filter_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = (self.input_width -
                             filter_width) / self.stride + 1
        self.output_height = (self.input_height -
                              filter_height) / self.stride + 1
        self.output_array = np.zeros((self.channel_number,
                                      self.output_height, self.output_width))
        network.append_layer(self)

    def forward(self, input_array):
        self.input_array = input_array
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d, i, j] = (
                        get_patch(input_array[d], i, j,
                                  self.filter_width,
                                  self.filter_height,
                                  self.stride).max())

    def calc_layer_delta(self, downstream_layer):
        self.delta_array = np.zeros(self.input_array.shape)
        sensitivity_array = downstream_layer.get_transformed_delta().reshape(self.output_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(
                        self.input_array[d], i, j,
                        self.filter_width,
                        self.filter_height,
                        self.stride)
                    k, l = get_max_index(patch_array)
                    self.delta_array[d,
                                     i * self.stride + k,
                                     j * self.stride + l] = \
                        sensitivity_array[d, i, j]

    def get_output(self):
        return self.output_array

    def get_transformed_delta(self):
        return self.delta_array

    def update_weight(self, rate):
        pass