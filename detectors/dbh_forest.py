#! /usr/bin/python
#
# Implemented by Jiayu Wang (email: jiayuw6@student.unimelb.edu.au). Copyright reserved.
#

import numpy as np
import copy as cp
import dbh_tree as dtree
import random
import time


class DBHForest:
    def __init__(self, num_trees, sampler, dbh_family, granularity=1):
        self._num_trees = num_trees
        self._sampler = sampler
        self._dbh_family = dbh_family #can be name of the distance functions
        self._granularity = granularity #used for local detection
        self._trees = []

    def fit(self, data):
        start_function_time = time.time()

        # Uncomment the following code for continuous values
        # indices = range(len(data))
        # data = np.c_[indices, data]

        self._trees = []
        self._sampler.fit(data)  #bagging.fit is in what math theory
        sampled_data = self._sampler.draw_samples(data) #a parameter given when creating an object, now it's 100

        # Generate hash functions in family for each tree
        dbh_instances = []
        # for i in range(self._num_trees):
            # Select DBH functions from the whole dataset for each tree
            # transformed_data = data
            # if self._sampler._bagging != None:
            #     transformed_data = self._sampler._bagging_instances[i].get_transformed_data(data)
            # self._dbh_family.fit(transformed_data)   #currently send all the data into the function builder, later need to try one subset and multiple subsets

            # # Select DBH functions from the subset for each tree
            # transformed_data = sampled_data[i]
            # if self._sampler._bagging != None:
            #     transformed_data = self._sampler._bagging_instances[i].get_transformed_data(transformed_data)
            # self._dbh_family.fit(transformed_data)
            # dbh_instances.append(cp.deepcopy(self._dbh_family.get_array()))  #family should be 50 groups of (X1,X2,t1,t2)

        # Generate hash functions in one time
        transformed_data = sampled_data[0] #function -> get 1 sample for generating (X1, X2, t1, t2) combinations
        if self._sampler._bagging != None:
            transformed_data = self._sampler._bagging_instances[0].get_transformed_data(transformed_data)
        self._dbh_family.fit(transformed_data)
        all_instances = cp.deepcopy(self._dbh_family.get_array())
        for i in range(self._num_trees):
            dbh_instances.append(random.sample(all_instances, 50))

        end_function_time = time.time()
        print "Have generated all combinations"
        print end_function_time - start_function_time

        # Build DBH trees in the forest
        for i in range(self._num_trees):
            tree = dtree.DBHTree(dbh_instances[i], self._dbh_family)
            tree.build(sampled_data[i])   #height problem: whether still need the height limitation?
            # tree.print_tree(tree.get_root(), ' ')
            self._trees.append(tree)
        end_build_time = time.time()
        print "build finished!"
        print end_build_time - end_function_time


    def decision_function(self, data):
        # Uncomment the following code for continuous data
        # indices = range(len(data))
        # data = np.c_[indices, data]

        depths = []
        data_size = len(data)
        for i in range(data_size):
            d_depths = []
            for j in range(self._num_trees):
                transformed_data = data[i]
                if self._sampler._bagging != None:
                    transformed_data = self._sampler._bagging_instances[j].get_transformed_data(np.mat(data[i])).A1
                d_depths.append(self._trees[j].predict(self._granularity, transformed_data))
            depths.append(d_depths)

        # Arithmatic mean
        avg_depths = []
        for i in range(data_size):
            depth_avg = 0.0
            for j in range(self._num_trees):
                depth_avg += depths[i][j]
            depth_avg /= self._num_trees
            avg_depths.append(depth_avg)

        avg_depths = np.array(avg_depths)
        # print "average depth list"
        # print avg_depths
        return -1.0 * avg_depths
