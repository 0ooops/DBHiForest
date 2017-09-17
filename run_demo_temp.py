#! /usr/bin/python
#
# Implemented by Xuyun Zhang (email: xuyun.zhang@auckland.ac.nz). Copyright reserved.
#

import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_auc_score
from detectors import VSSampling
from detectors import dbh_forest
from detectors import dbh

# rng = np.random.RandomState(42)
# number of trees in forest, and the number of samples also uses this
num_ensemblers=100

data = pd.read_csv('dat/glass.csv', header=None)
# data = pd.read_csv('dat/ionosphere.csv', header=None)
# data = pd.read_csv('dat/breastw.csv', header=None)
# data = pd.read_csv('dat/pima.csv', header=None)
# data = pd.read_csv('dat/vowel-train2.csv', header=None)
# data = pd.read_csv('dat/two-spiral.csv', header=None)
# data = pd.read_csv('dat/two-spiral-surrounded.csv', header=None)
# data = pd.read_csv('dat/two-spiral-local.csv', header=None)
# data = pd.read_csv('dat/local-one-anomaly.csv', header=None)

X = data.as_matrix()[:, :-1].tolist()
ground_truth = data.as_matrix()[:, -1].tolist()

classifiers = [
               # ("L1SH", LSHForest(num_ensemblers, VSSampling(num_ensemblers), E2LSH(norm=1))),
               ("DBH", dbh_forest.DBHForest(num_ensemblers, VSSampling(num_ensemblers), dbh.DBH())),
               # ("DBH", dbh_forest.DBHForest(num_ensemblers, VSSampling(num_ensemblers),
               #                              dbh.DBH(distance_function="Euclidean"))),
               # ("DBH", dbh_forest.DBHForest(num_ensemblers, VSSampling(num_ensemblers),
               #                              dbh.DBH(distance_function="SquaredEuclidean"))),
               # ("DBH", dbh_forest.DBHForest(num_ensemblers, VSSampling(num_ensemblers),
               #                              dbh.DBH(distance_function="BrayCurtis"))),
               # ("DBH", dbh_forest.DBHForest(num_ensemblers, VSSampling(num_ensemblers),
               #                              dbh.DBH(distance_function="Canberra"))),
               # ("DBH", dbh_forest.DBHForest(num_ensemblers, VSSampling(num_ensemblers),
               #                              dbh.DBH(distance_function="Chebyshev"))),
               # ("DBH", dbh_forest.DBHForest(num_ensemblers, VSSampling(num_ensemblers),
               #                              dbh.DBH(distance_function="Correlation"))),
               ("DBH", dbh_forest.DBHForest(num_ensemblers, VSSampling(num_ensemblers),
                                            dbh.DBH(distance_function="Cosine")))]

# file = open('./result_temp.csv','w')
for i, (clf_name, clf) in enumerate(classifiers):
    print "	" + clf_name + ":"
    # file.write(str(clf_name))
    start_time = time.time()
    clf.fit(X)
    train_time = time.time() - start_time
    y_pred = clf.decision_function(X).ravel()
    test_time = time.time() - start_time - train_time
    auc = roc_auc_score(ground_truth, -1.0 * y_pred)
    print "AUC: ", auc
    print "Training time:	", train_time
    print "Testing time: ", test_time
    # file.write("AUC: " + str(auc))
    # file.write("Training time: " + str(train_time))
    # file.write("Testing time: " + str(test_time))
# file.close()