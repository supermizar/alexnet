class Network(object):
    def __init__(self):
        self.layers = []

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

    def train_one_sample(self, label, sample, rate):
        """
        train network with one sample
        """
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label):
        """
        calc delta of each layer
        """
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        """
        update weights of each connection or filter
        """
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

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

    def predict(self, sample):
        """
        predict output according to input
        """
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    def dump(self):
        """
        print network info
        """
        for layer in self.layers:
            layer.dump()
