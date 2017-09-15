#! /usr/bin/python
#
# Implemented by Jiayu Wang (email: jiayuw6@student.unimelb.edu.au). Copyright reserved.
#

import scipy.spatial.distance as cal_dis
import random
import math
import numpy as np


class DBH:
    def __init__(self, default_pool_size=500, distance_function="Manhattan"):
        self._default_pool_size = default_pool_size
        self._distance_function = distance_function
        self.DBH_array = None

    def get_func(self):
        return self._distance_function

    def fit(self, data):
        if len(data) == 0:
            return

        # self._dimensions = len(data[0]) - 1
        self.DBH_array = []
        length = len(data)

        for i in range(self._default_pool_size):
            data_copy = data
            dis_array = []

            # Select two random points from dataset
            x_1 = random.randint(0, length - 1)
            x_2 = random.randint(0, length - 1)
            while x_2 == x_1:
                x_2 = random.randint(0, length - 1)
            X1 = data_copy[x_1]
            X2 = data_copy[x_2]

            # Calculate values for all the other data based on "line projection" function defined by X1 and X2
            for j in range(length):
                if j != x_1 and j != x_2:
                    dis = self.line_projection(self._distance_function, X1, X2, data_copy[j])
                    if dis is not None:
                        dis_array.append(dis)
                    else:
                        print "Invalid line projection value"
            dis_array.sort()

            # Randomly select t1 and calculate t2
            # need to discuss or test whether randomly generating is proper, whether need to consider the normal distribution etc.
            t_1 = random.uniform(dis_array[0], dis_array[len(dis_array) - 1])
            t_2 = self.cal_t(dis_array, t_1)
            if t_2 < t_1:
                new_function = (X1, X2, t_2, t_1)
            else:
                new_function = (X1, X2, t_1, t_2)
            if new_function not in self.DBH_array:
                self.DBH_array.append(new_function)
        self.DBH_array = np.array(self.DBH_array)

    def cal_t(self, array, t1):
        # print array
        # print t1
        left_count = 0
        for i in array:
            if i < t1:
                left_count += 1
        if left_count < len(array) / 2:
            t2 = random.uniform(array[int(len(array) / 2) + left_count], array[len(array) - 1])
        elif left_count > len(array) / 2:
            t2 = random.uniform(array[0], array[left_count - int(len(array) / 2)])
        else:
            index = random.randint(0, 1)
            if index == 0:
                t2 = float('inf')
            else:
                t2 = float('-inf')
        return t2

    def get_array(self):
        return self.DBH_array

    @staticmethod
    def line_projection(distance_func, x1, x2, x):
        if distance_func == "Manhattan":
            return (math.pow(cal_dis.cityblock(x, x1), 2) + math.pow(cal_dis.cityblock(x1, x2), 2)
                      - math.pow(cal_dis.cityblock(x, x2), 2)) / (2.0 * cal_dis.cityblock(x1, x2))
        if distance_func == "Euclidean":
            return (math.pow(cal_dis.euclidean(x, x1), 2) + math.pow(cal_dis.euclidean(x1, x2), 2)
                      - math.pow(cal_dis.euclidean(x, x2), 2)) / (2.0 * cal_dis.euclidean(x1, x2))
        if distance_func == "SquaredEuclidean":
            return (math.pow(cal_dis.sqeuclidean(x, x1), 2) + math.pow(cal_dis.sqeuclidean(x1, x2), 2)
                      - math.pow(cal_dis.sqeuclidean(x, x2), 2)) / (2.0 * cal_dis.sqeuclidean(x1, x2))
        if distance_func == "BrayCurtis":
            return (math.pow(cal_dis.braycurtis(x, x1), 2) + math.pow(cal_dis.braycurtis(x1, x2), 2)
                      - math.pow(cal_dis.braycurtis(x, x2), 2)) / (2.0 * cal_dis.braycurtis(x1, x2))
        if distance_func == "Canberra":
            return (math.pow(cal_dis.canberra(x, x1), 2) + math.pow(cal_dis.canberra(x1, x2), 2)
                      - math.pow(cal_dis.canberra(x, x2), 2)) / (2.0 * cal_dis.canberra(x1, x2))
        if distance_func == "Chebyshev":
            return (math.pow(cal_dis.chebyshev(x, x1), 2) + math.pow(cal_dis.chebyshev(x1, x2), 2)
                      - math.pow(cal_dis.chebyshev(x, x2), 2)) / (2.0 * cal_dis.chebyshev(x1, x2))
        if distance_func == "Correlation":
            return (math.pow(cal_dis.correlation(x, x1), 2) + math.pow(cal_dis.correlation(x1, x2), 2)
                      - math.pow(cal_dis.correlation(x, x2), 2)) / (2.0 * cal_dis.correlation(x1, x2))
        if distance_func == "Cosine":
            return (math.pow(cal_dis.cosine(x, x1), 2) + math.pow(cal_dis.cosine(x1, x2), 2)
                      - math.pow(cal_dis.cosine(x, x2), 2)) / (2.0 * cal_dis.cosine(x1, x2))
        return None