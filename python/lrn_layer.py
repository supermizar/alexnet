import numpy as np


class LrnLayer(object):
    def __init__(self, network, k, alpha, n, beta):
        self.input_array = network.layers[-1].get_output()
        self.k = k
        self.alpha = alpha
        self.n = n
        self.beta = beta
        self.output_array = np.zeros(self.input_array.shape)
        network.append_layer(self)

    def forward(self, input_array):
        self.input_array = input_array
        self.normalize_matrix = np.zeros(input_array.shape)
        for i in range(0, input_array.shape[0]):
            sum_matrix = np.zeros([input_array.shape[1], input_array.shape[2]])
            for j in range(max(0, i-self.n/2), min(self.input_array.shape[0]-1, i+self.n/2)):
                sum_matrix += input_array[j,:,:] ** 2
            self.normalize_matrix[i, :, :] = self.k + self.alpha * sum_matrix
        self.divisor_matrix = self.normalize_matrix ** (-self.beta)
        self.output_array = self.input_array * self.divisor_matrix

    def get_output(self):
        return self.output_array

    def calc_layer_delta(self, downstream_layer):
        delta_array_downstream = downstream_layer.get_transformed_delta()
        self.delta_array = np.zeros(downstream_layer.get_transformed_delta().shape)
        for i in range(0, self.delta_array.shape[0]):
            sum_matrix = np.zeros([self.delta_array.shape[1], self.delta_array.shape[2]])
            for j in range(max(0, i - self.n / 2), min(self.input_array.shape[0] - 1, i + self.n / 2)):
                sum_matrix += self.output_array[j,:,:] * delta_array_downstream[j,:,:] / self.normalize_matrix[j,:,:]
            self.delta_array[i,:,:] = self.divisor_matrix[i,:,:] * delta_array_downstream[i,:,:] - \
                                      2 * self.alpha * self.beta * self.input_array[i,:,:] * sum_matrix

    def update_weight(self, rate):
        pass

    def get_transformed_delta(self):
        return self.delta_array