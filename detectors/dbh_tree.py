#! /usr/bin/python
#
# Implemented by Jiayu Wang (email: jiayuw6@student.unimelb.edu.au). Copyright reserved.
#

import numpy as np
import dbh_node as nd
import dbh


class DBHTree:
    def __init__(self, dbh_instances, distance_func):
        self._dbh_instances = dbh_instances
        self._height_limit = 0
        self._distance_func = distance_func
        self._root = None
        self._n_samples = 0

    def build(self, data):
        self._n_samples = len(data)
        if self._n_samples == 0:
            return None
        self._height_limit = self.get_random_height(self._n_samples)
        current_height = 0
        self._root = self.recur_build(data, current_height)

    def recur_build(self, data, current_height):
        if len(data) == 0:
            return None
        if len(data) == 1 or current_height > self._height_limit:
            return nd.DBHNode(current_height, None, None, len(data))
        left = []
        right = []
        func = self._dbh_instances[current_height]
        for i in data:
            dis = dbh.DBH.line_projection(self._distance_func.get_func(), func[0], func[1], i)
            if dis >= func[2] and dis <= func[3]:
                left.append(i)
            else:
                right.append(i)
        return nd.DBHNode(current_height, self.recur_build(left, current_height+1), self.recur_build(right, current_height+1), len(data))

    def print_tree(self, node, leftStr=' '):
        if node is None:
            return
        print leftStr + '(' + str(len(leftStr)) + ',' + str(node.get_index()) + '):' + str(
            node.get_size())
        if node.get_left_child() != None:
            self.print_tree(node.get_left_child(), leftStr + ' ')
        if node.get_right_child() != None:
            self.print_tree(node.get_right_child(), leftStr + ' ')

    def get_root(self):
        return self._root

    def predict(self, granularity, data_point):
        current_height = 0
        path_length = self.get_path_length(self._root, self._height_limit, current_height, granularity, data_point)
        return pow(2.0, (-1.0 * path_length / self.get_unsuccess_average(self._n_samples)))

    def get_path_length(self, node, height_limit, current_height, granularity, data_point):
        if node is None:
            return -1
        children = node.get_children()
        if children is None or current_height > height_limit:
            real_depth = node.get_index()
            return current_height * np.power(1.0 * real_depth / max(current_height, 1.0), granularity) \
                   + self.get_unsuccess_average(node.get_size())
        else:
            func = self._dbh_instances[current_height]
            dis = dbh.DBH.line_projection(self._distance_func.get_func(), func[0], func[1], data_point)
            if dis >= func[2] and dis <= func[3]:
                return self.get_path_length(node.get_left_child(), height_limit, current_height+1, granularity, data_point)
            else:
                return self.get_path_length(node.get_right_child(), height_limit, current_height+1, granularity, data_point)

    @staticmethod
    def get_unsuccess_average(sample_size):
        if sample_size < 2:
            return 0
        elif sample_size == 2:
            return 1
        else:
            return 2 * (np.log10(sample_size - 1) + 0.5772156649) - 2 * (sample_size - 1) / sample_size

    @staticmethod
    def get_random_height(num_samples):
        return 2 * np.log2(num_samples) + 0.8327

