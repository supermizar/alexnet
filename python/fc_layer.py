import math
from functools import reduce
import random


class FcLayer(object):
    def __init__(self, network, layer_index, node_count):
        '''
        init one layer
        layer_index: index of the layer
        node_count: sum of nodes in this layer
        '''
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))
        network.append_layer(self)

    def set_output(self, data):
        '''
        set output of the layer, used when self is output layer
        '''
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        '''
        calculate the output vector
        '''
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        '''
        print layer info
        '''
        for node in self.nodes:
            print node


class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        '''
        init the connection, initial weight should be a random small number
        '''
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        '''
        calculate gradient
        '''
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def get_gradient(self):
        '''
        get current gradient
        '''
        return self.gradient

    def update_weight(self, rate):
        '''
        update weight according to gradient descending algorithm
        '''
        self.calc_gradient()
        self.weight += rate * self.gradient

    def __str__(self):
        '''
        print connection info
        '''
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)


class ConstNode(object):
    def __init__(self, layer_index, node_index):
        '''
        layer_index: layer index of node belong to
        node_index: node index of current layer
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self, conn):
        '''
        append a connection to downstream
        '''
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        '''
        calculate delta when node belong to hidden
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        '''
        print const node info
        '''
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


class Node(object):
    def __init__(self, layer_index, node_index):
        '''
        layer_index: layer index of node belong to
        node_index: node index of current layer
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        '''
        set the output, used when node belong to output layer
        '''
        self.output = output

    def append_downstream_connection(self, conn):
        '''
        append a connection to downstream
        '''
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        '''
        append a connection to upstream
        '''
        self.upstream.append(conn)

    def calc_output(self):
        '''
        calculate output of current layer
        '''
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = self.sigmoid(output)

    def calc_hidden_layer_delta(self):
        '''
        calc delta when layer is hidden layer
        '''
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        '''
        calc delta when layer is output layer
        '''
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        '''
        print node info
        '''
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
