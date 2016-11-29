import numpy as np


class ConvLayer(object):
    def __init__(self, network, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, filter_number,
                 zero_padding, stride, activator,
                 learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = \
            ConvLayer.calculate_output_size(
                self.input_width, filter_width, zero_padding,
                stride)
        self.output_height = \
            ConvLayer.calculate_output_size(
                self.input_height, filter_height, zero_padding,
                stride)
        self.output_array = np.zeros((self.filter_number,
                                      self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width,
                                       filter_height, self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate
        network.append_layer(self)

    def bp_sensitivity_map(self, sensitivity_array,
                           activator):
        """
        calculate sensitivity map passed to upstream layer
        sensitivity_array: sensitivity map of current layer
        activator: activator of upstream layer
        """
        # handle stride, expand sensitivity map
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
        # full conv, do zp to sensitivity map
        # though zp has epsilon, it won't take into consider cause that won't passed upstream
        expanded_width = expanded_array.shape[2]
        zp = (self.input_width +
              self.filter_width - 1 - expanded_width) / 2
        padded_array = padding(expanded_array, zp)
        # init delta_array, for saving sensitivity map passed to upstream
        self.delta_array = self.create_delta_array()
        # for those conv layer has multi filters, sensitivity map passed to upstream finally, is sum of
        # all filter's sensitivity map
        for f in range(self.filter_number):
            filter = self.filters[f]
            # rotate filter 180 degree
            flipped_weights = np.array(map(
                lambda i: np.rot90(i, 2),
                filter.get_weights()))
            # calc a filter for delta_array
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d],
                     delta_array[d], 1, 0)
            self.delta_array += delta_array
        # calc element-wise of output and activator's bias-derivation
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array,
                        activator.backward)
        self.delta_array *= derivative_array

    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        # calc sensitivity map's size
        # calc size of sensitivity map when stride is 1
        expanded_width = (self.input_width -
                          self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height -
                           self.filter_height + 2 * self.zero_padding + 1)
        # construct new sensitivity_map
        expand_array = np.zeros((depth, expanded_height,
                                 expanded_width))
        # copy error from snesitivity map
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = \
                    sensitivity_array[:, i, j]
        return expand_array

    def create_delta_array(self):
        return np.zeros((self.channel_number,
                         self.input_height, self.input_width))

    def bp_gradient(self, sensitivity_array):
        # handle stride, expand initial sensitivity map
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
        for f in range(self.filter_number):
            # calc gradient of each weight
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d],
                     expanded_array[f],
                     filter.weights_grad[d], 1, 0)
            # calc the gradient of bias
            filter.bias_grad = expanded_array[f].sum()

    def update(self):
        """
        update weight according to gradient descend
        """
        for filter in self.filters:
            filter.update(self.learning_rate)

    @staticmethod
    def calculate_output_size(input_size,
                              filter_size, zero_padding, stride):
        return (input_size - filter_size +
                2 * zero_padding) / stride + 1

    def forward(self, input_array):
        """
        calc output of conv layer
        result saved in self.output_array
        """
        self.input_array = input_array
        self.padded_input_array = padding(input_array,
                                          self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array,
                 filter.get_weights(), self.output_array[f],
                 self.stride, filter.get_bias())
        element_wise_op(self.output_array,
                        self.activator.forward)

    def get_output(self):
        return self.output_array


# do element wise operation to numpy array
def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)


def conv(input_array,
         kernel_array,
         output_array,
         stride, bias):
    """
    calculate convolution for 2D 3D
    """
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (
                                     get_patch(input_array, i, j, kernel_width,
                                               kernel_height, stride) * kernel_array
                                 ).sum() + bias


def get_patch(input, i, j, kernel_width, kernel_height, stride):
    pass


def padding(input_array, zp):
    """
    add zero padding for 2D 3D
    """
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((
                input_depth,
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[:,
            zp: zp + input_height,
            zp: zp + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[zp: zp + input_height,
            zp: zp + input_width] = input_array
            return padded_array


class Filter(object):
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4, 1e-4,
                                         (depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(
            self.weights.shape)
        self.bias_grad = 0

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (
            repr(self.weights), repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad


class ReluActivator(object):
    def forward(self, weighted_input):
        # return weighted_input
        return max(0, weighted_input)

    def backward(self, output):
        return 1 if output > 0 else 0
