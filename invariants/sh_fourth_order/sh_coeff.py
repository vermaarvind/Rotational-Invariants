"""
Spherical Harmonics Coeffecients as features 

"""
# standard imports
import cmath
import math
import pandas
import os
import sys
import numpy as np
import scipy.io
import nrrd
import random
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from activation_functions import sigmoid_function, tanh_function, linear_function, LReLU_function, ReLU_function, elliot_function, symmetric_elliot_function, softmax_function
from cost_functions import sum_squared_error
from learning_algorithms import backpropagation, scaled_conjugate_gradient, scipyoptimize, resilient_backpropagation
from neuralnet import NeuralNet
from tools import Instance
import datetime
import time

####################___Loading the train ODF .nrrd file____###########################

train_data_1, options = nrrd.read('one-train-odfs.nrrd')
train_data_2, options = nrrd.read('two-train-odfs.nrrd')
train_data_3, options = nrrd.read('three-train-odfs.nrrd')

train1 = pandas.DataFrame(train_data_1)
train2 = pandas.DataFrame(train_data_2)
train3 = pandas.DataFrame(train_data_3)

df_train_1 = train1.transpose()
df_train_2 = train2.transpose()
df_train_3 = train3.transpose()


df_train_1.loc[:, 'y'] = 0
df_train_2.loc[:, 'y'] = 1
df_train_3.loc[:, 'y'] = 2

df_train = pandas.concat([df_train_1, df_train_2, df_train_3])

#print "training data : ", df_train.shape

####################___Loading the test ODF .nrrd file____###########################

test_data_1, options = nrrd.read('one-test-odfs.nrrd')
test_data_2, options = nrrd.read('two-test-odfs.nrrd')
test_data_3, options = nrrd.read('three-test-odfs.nrrd')

test1 = pandas.DataFrame(test_data_1)
test2 = pandas.DataFrame(test_data_2)
test3 = pandas.DataFrame(test_data_3)

df_test_1 = test1.transpose()
df_test_2 = test2.transpose()
df_test_3 = test3.transpose()


df_test_1.loc[:, 'y'] = 0
df_test_2.loc[:, 'y'] = 1
df_test_3.loc[:, 'y'] = 2

df_test = pandas.concat([df_test_1, df_test_2, df_test_3])

#print "test data : ", df_test.shape

###############################################################################################

y_train = df_train["y"].values
df_train = df_train.drop(["y"], 1)
X_train = df_train.values


y_test = df_test["y"].values
df_test = df_test.drop(["y"], 1)
X_test = df_test.values

###############################################################################################

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
    print "The minimum number of trees are ", best_par
              

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
    train_score = dlf.score(X_train, y_train)     #(why it is test here, it should be train here instead of test)
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
    print "Test time : ", time_test
    print "--------------------------------------------------------"    
    print "Test score : ", test_score
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
        "n_inputs"              : 15,       # Number of network input signals
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
time_list_RF, time_train_predict_RF, cm_RF  = RandomForest_cross_validation(X_train, y_train, X_test, y_test)


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
df_res.to_excel("sh_coeff.xlsx")
df_res
'''
