import time
import numpy as np
indent = '   '
class Network(object):
    def __init__(self):
        self.layers = []
        self.showlog = False

    def append_layer(self, layer):
        self.layers.append(layer)

    def train(self, labels, data_set, rate, iteration):
        """
        train the network
        labels: labels of samples
        data_set: n-dim array, n-1 dim indicate the sample dim, last dim means the num of samples
        """
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate, showlog=True):
        """
        train network with one sample
        """
        self.showlog=showlog
        self.indexes = np.array([x+1 for x in range(0, len(self.layers))])
        if showlog:
            print 'Training with one sample started'
            print indent + 'Procedure predict started'
        clk1 = time.time()
        self.predict(sample, training=True)
        clk2 = time.time()
        if showlog:
            print indent + 'Procedure predict finished, time cost(s):' + str(clk2 - clk1)
            print indent + 'Procedure delta calculation started'
        self.calc_delta(label)
        clk3 = time.time()
        if showlog:
            print indent + 'Procedure delta calculation finished, time cost(s):' + str(clk3 - clk2)
            print indent + 'Procedure weight update started'
        self.update_weight(rate)
        clk4 = time.time()
        if showlog:
            print indent + 'Procedure weight update finished, time cost(s):' + str(clk4 - clk3)
            print 'Training with one sample finished'

    def calc_delta(self, label):
        """
        calc delta of each layer
        """
        self.delta_calc_time=[]
        output_layer = self.layers[-1]
        layer_index = len(self.layers)
        clk1 = time.time()
        output_layer.calc_output_layer_delta(label)
        clk2 = time.time()
        if self.showlog:
            print 2*indent + "Delta calculation of layer " + str(layer_index) + " finished, time cost(s): " + str(clk2 - clk1)
        self.delta_calc_time.insert(0, clk2-clk1)
        downstream_layer = output_layer
        for layer in self.layers[-2::-1]:
            layer_index -= 1
            clk1 = time.time()
            layer.calc_layer_delta(downstream_layer)
            clk2 = time.time()
            if self.showlog:
                print 2*indent + "Delta calculation of layer " + str(layer_index) + " finished, time cost(s): " + str(clk2 - clk1)
            self.delta_calc_time.insert(0, clk2-clk1)
            downstream_layer = layer

    def update_weight(self, rate):
        """
        update weights of each connection or filter
        """
        self.update_weight_time=[]
        layer_index = 1
        for layer in self.layers:
            clk1 = time.time()
            layer.update_weight(rate)
            clk2 = time.time()
            if self.showlog:
                print 2*indent + "Weight update of layer " + str(layer_index) + " finished, time cost(s): " + str(clk2 - clk1)
            self.update_weight_time.append(clk2 - clk1)
            layer_index += 1

    def calc_gradient(self):
        """
        calc gradients of each connection or filter
        """
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        """
        get every gradient of connections under a network, only for gradient check
        """
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample, training=False):
        """
        predict output according to input
        """
        self.predict_time=[]
        layer_index = 1
        clk1 = time.time()
        self.layers[0].forward(sample, training=training)
        clk2 = time.time()
        self.predict_time.append(clk2 - clk1)
        if training and self.showlog:
            print 2*indent + "Forward calculation of layer " + str(layer_index) + " finished, time cost(s): " + str(clk2 - clk1)
        for i in range(1, len(self.layers)):
            layer_index += 1
            clk1 = time.time()
            self.layers[i].forward(self.layers[i-1].get_output(), training=training)
            clk2 = time.time()
            if training and self.showlog:
                print 2*indent + "Forward calculation of layer " + str(layer_index) + " finished, time cost(s): " + str(clk2 - clk1)
            self.predict_time.append(clk2 - clk1)
        if not training and self.showlog:
            print "***Predict result based on alexnet model***"
            print self.layers[-1].get_output()
        return self.layers[-1].get_output()

    def dump(self):
        """
        print network info
        """
        for layer in self.layers:
            layer.dump()
