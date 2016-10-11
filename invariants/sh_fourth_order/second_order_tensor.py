"""
Eigen vlaues of second order tensors as features

"""

import cmath
import math
import pandas
import os
import sys
import datetime
import time
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy.io
from matplotlib import pyplot as pot
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import nrrd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from numpy import linalg as LA
from activation_functions import sigmoid_function, tanh_function, linear_function, LReLU_function, ReLU_function, elliot_function, symmetric_elliot_function, softmax_function
from cost_functions import sum_squared_error
from learning_algorithms import backpropagation, scaled_conjugate_gradient, scipyoptimize, resilient_backpropagation
from neuralnet import NeuralNet
from tools import Instance

####################___Loading the ODF .nrrd file____###########################

def reform_res(row):

        return np.asarray([[row[0], row[1], row[2]], 
                           [row[1], row[3], row[4]], 
                           [row[2], row[4], row[5]]])
    
    
# create the function for data transformations

def load_transform(file_path, l):

    n_x, options = nrrd.read(file_path)

    z = n_x.transpose()
    a = np.array(z)
    data = a[:,:6]

    m = [ [1.1816359006036771806, 0, 0, 1.1816359006036771806, 0, 1.1816359006036771806],
          [0.9152912328637688999, 0, 0, -0.9152912328637688999, 0, 0],
          [0, 0, -1.8305824657275377998, 0, 0, 0],
          [-0.52844363968080143579, 0, 0, -0.52844363968080143579, 0, 1.0568872793616028716],
          [0, 0, 0, 0, -1.8305824657275377998, 0],
          [0, 1.8305824657275377998, 0, 0, 0, 0] ]

    m = np.asarray(m)

    res = np.dot(data, m)   

    res_final = np.asarray([reform_res(row) for row in res])
    
    eigen_val = np.asarray([LA.eigvals(item) for item in res_final])

    # add the label class
    
    labels = [l] * len(eigen_val)
    
    return eigen_val, list(labels)

# read all datasets
X_train_1, y_train_1 = load_transform('one-train-odfs.nrrd', 0)
X_train_2, y_train_2 = load_transform('two-train-odfs.nrrd', 1)
X_train_3, y_train_3 = load_transform('three-train-odfs.nrrd', 2)
X_test_1, y_test_1 = load_transform('one-test-odfs.nrrd', 0)
X_test_2, y_test_2 = load_transform('two-test-odfs.nrrd', 1)
X_test_3, y_test_3 = load_transform('three-test-odfs.nrrd', 2)

# merge datasets
X_train_temp = np.append(X_train_1, X_train_2, axis=0)
X_train = np.append(X_train_temp, X_train_3, axis=0)

y_train_temp = np.append(y_train_1, y_train_2, axis=0)
y_train = np.append(y_train_temp, y_train_3, axis=0)

X_test_temp = np.append(X_test_1, X_test_2, axis=0)
X_test = np.append(X_test_temp, X_test_3, axis=0)

y_test_temp = np.append(y_test_1, y_test_2, axis=0)
y_test = np.append(y_test_temp, y_test_3, axis=0)

def RandomForest_cross_validation(X_train, y_train, X_test, y_test):
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"   
    print "Random Forest Classifier"
    print "--------------------------------------------------------"
    print "Grid search"
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"

    # grid search by the best parameters
    parameters = []
    acc_all = []
    best_acc = 0
    best_par = 0
    time_list_RF = []
    for n in [10, 100, 200, 500, 1000]: 
        timeStart = time.time()
        dlf = RandomForestClassifier(n_estimators=n, bootstrap=True, 
                            criterion="entropy", max_depth=None, max_features='auto',
                            max_leaf_nodes=None, min_samples_leaf=1,
                            min_samples_split=2,  n_jobs=1,
                            oob_score=False, random_state=0, verbose=0)
        dlf.fit(X_train, y_train)
        acc = dlf.score(X_test, y_test)
        
        print "Score = ", acc
        print "Number of trees =  ", n

                
        parameters.append(n)
        acc_all.append(acc)
                
        if acc > best_acc:
                    best_acc = acc
                    best_par = n
                    
        if len(acc_all) >=3:
            if acc < acc_all[-2] or acc == acc_all[-3]:
                break
                
        time_list_RF.append([n, acc, time.time() - timeStart])
                
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print dlf
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"    
    
    #print "The best Accuracy is ", best_acc
    print "The best minimum number of trees is ", best_par
              
    time_train_predict = []  
           
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print "Training and Prediction"
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"


    dlf = RandomForestClassifier(n_estimators=best_par, bootstrap=True, 
                            criterion="entropy", max_depth=None, max_features='auto',
                            max_leaf_nodes=None, min_samples_leaf=1,
                            min_samples_split=2,  n_jobs=1,
                            oob_score=False, random_state=0, verbose=0)

    #train
    timeStart_train = time.time()
    dlf.fit(X_train, y_train)
    train_score = dlf.score(X_test, y_test)
    time_train = time.time() - timeStart_train

    #Predict
    timeStart_test = time.time()
    pred = dlf.predict(X_test)
    test_score = dlf.score(X_test, y_test)
    time_test = time.time() - timeStart_test
        
    time_train_predict.append([n, time.time() - timeStart_train, train_score, time_test, test_score])      

    cm = confusion_matrix(y_test, pred)
    
    #print "Training score : ", train_score
    print "Training time :", time_train 
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"    
    print "Test score : ", test_score
    print "Test time : ", time_test
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print "Confusion matrix : "
    print cm
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    
    return time_list_RF, time_train_predict, cm 
    
    
def KNeighbors(X_train, y_train, X_test, y_test):
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print "kNN Classifier"
    print "--------------------------------------------------------"
    print "Grid search"
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"

    # grid search by the best parameters
    parameters = []
    acc_all = []
    best_acc = 0
    best_par = 0
    time_list_KN = []
    for n in [5, 10, 20, 30, 50]:
        timeStart = time.time()
        dlf = KNeighborsClassifier(n_neighbors=n, leaf_size=30, p=2, metric='minkowski', metric_params=None)
        dlf.fit(X_train, y_train)
        acc_avg = dlf.score(X_test, y_test)

        print "Score = ", acc_avg
        print "Number of neighbors =  ", n



        parameters.append(n)
        acc_all.append(acc_avg)

        if acc_avg > best_acc:
                        best_acc = acc_avg
                        best_par = n

        if len(acc_all) >=3:
                if acc_avg < acc_all[-2] or acc_avg == acc_all[-3]:
                    break
        time_list_KN.append([n, acc_avg, time.time() - timeStart])
    
    
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print dlf
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"    

    #print "The best Accuracy is ", best_acc
    print "The best minimum number of neighbors is ", best_par
    
    
    time_train_predict = []  
    
    
        
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print "Training and Prediction"
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"


    dlf = KNeighborsClassifier(n_neighbors=best_par, leaf_size=30, p=2, metric='minkowski', metric_params=None)

    #train
    timeStart_train = time.time()
    dlf.fit(X_train, y_train)
    train_score = dlf.score(X_test, y_test)
    time_train = time.time() - timeStart_train

    #Predict
    timeStart_test = time.time()
    pred = dlf.predict(X_test)
    test_score = dlf.score(X_test, y_test)
    time_test = time.time() - timeStart_test
        
    time_train_predict.append([n, time.time() - timeStart_train, train_score, time_test, test_score])      

    cm = confusion_matrix(y_test, pred)
    
    #print "Training score : ", train_score
    print "Training time :", time_train 
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"    
    print "Test score : ", test_score
    print "Test time : ", time_test
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print "Confusion matrix : "
    print cm
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    
    return time_list_KN, time_train_predict, cm 

'''
def FCNN(X_train, y_train, X_test, y_test):
    
    training_one = []
    for i in xrange(len(X_train)):
        training_one.append(Instance(X_train[i], [y_train[i]]))

        
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print "Defining Neural Network"
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    
    settings = {
        # Required settings
        "cost_function"         : sum_squared_error,
        "n_inputs"              : 3,       # Number of network input signals
        "layers"                : [ (3, tanh_function), (1, sigmoid_function) ],
                                            # [ (number_of_neurons, activation_function) ]
                                            # The last pair in you list describes the number of output signals

        # Optional settings
        "weights_low"           : -0.1,     # Lower bound on initial weight range
        "weights_high"          : 0.1,      # Upper bound on initial weight range
        "save_trained_network"  : False,    # Whether to write the trained weights to disk

        #"input_layer_dropout"   : 0.000000001,      # dropout fraction of the input layer
        #"hidden_layer_dropout"  : 0.0,      # dropout fraction in all hidden layers
    }


    # initialize the neural network
    network = NeuralNet( settings )

    # load a stored network configuration
    # network = NeuralNet.load_from_file( "trained_configuration.pkl" )

    # Train the network using SciPy
    scipyoptimize(
            network,
            training_one, 
            method = "Newton-CG",
            ERROR_LIMIT = 1e-2
        )
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print "Fully connected Neural Network building"
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    
    network.print_test( training_one )
      
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"

'''
time_list_RF, time_train_predict_RF, cm_RF = RandomForest_cross_validation(X_train, y_train, X_test, y_test)

time_list_KN, time_train_predict_KN, cm_KN = KNeighbors(X_train, y_train, X_test, y_test)

#FCNN(X_train, y_train, X_test, y_test)

#Saving the results in a excel file
'''
df_res = pandas.DataFrame()

df_res.loc["Training Time (sec)", "RF ({} trees)".format(time_train_predict_RF[0][0])] = time_train_predict_RF[0][1]
df_res.loc["Training Accuracy", "RF ({} trees)".format(time_train_predict_RF[0][0])] = time_train_predict_RF[0][2]
df_res.loc["Prediction Time (sec)", "RF ({} trees)".format(time_train_predict_RF[0][0])] = time_train_predict_RF[0][3]
df_res.loc["Prediction Accuracy", "RF ({} trees)".format(time_train_predict_RF[0][0])] = time_train_predict_RF[0][4]

df_res.loc["Confusion matrix", "RF ({} trees)".format(time_train_predict_RF[0][0])] = str(cm_RF)



df_res.loc["Training Time (sec)", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = time_train_predict_KN[0][1]
df_res.loc["Training Accuracy", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = time_train_predict_KN[0][2]
df_res.loc["Prediction Time (sec)", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = time_train_predict_KN[0][3]
df_res.loc["Prediction Accuracy", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = time_train_predict_KN[0][4]

df_res.loc["Confusion matrix", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = str(cm_KN)


df_res = df_res.fillna('')
df_res.to_excel("second_order_tensor.xlsx")
df_res

'''



