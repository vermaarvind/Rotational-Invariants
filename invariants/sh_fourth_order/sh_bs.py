
"""

SH bispectrum features

"""

import cmath
import math
import pandas
import os
import sys
import numpy as np
import scipy.io
from matplotlib import pyplot as pot
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from activation_functions import sigmoid_function, tanh_function, linear_function, LReLU_function, ReLU_function, elliot_function, symmetric_elliot_function, softmax_function
from cost_functions import sum_squared_error
from learning_algorithms import backpropagation, scaled_conjugate_gradient, scipyoptimize, resilient_backpropagation
from neuralnet import NeuralNet
from tools import Instance
import nrrd
import datetime
import time
from sklearn.metrics import confusion_matrix

####################___Loading the ODF .mat file____###########################

def tijk_3d_sym_to_esh(ten):
    
    m =  [ [0.70898154036220639718, 0, 0, 1.4179630807244127944, 0, 
            1.4179630807244127944, 0, 0, 0, 0, 
            0.70898154036220639718, 0, 1.4179630807244127944, 0, 0.70898154036220639718],
           [0.78453534245465905705, 0, 0, 0, 0, 
            0.78453534245465905705, 0, 0, 0, 0, 
           -0.78453534245465905705, 0, -0.78453534245465905705, 0, 0],
           [0, 0, -1.5690706849093181141, 0, 0, 
            0, 0, -1.5690706849093181141, 0, -1.5690706849093181141, 
            0, 0, 0, 0, 0],
          [-0.45295169115497263546, 0, 0, -0.90590338230994527091, 0, 
            0.45295169115497263546, 0, 0, 0, 0, 
            -0.45295169115497263546, 0, 0.45295169115497263546, 0, 0.90590338230994527091],
           [0, 0, 0, 0, -1.5690706849093181141, 
            0, 0, 0, 0, 0, 
            0, -1.5690706849093181141, 0, -1.5690706849093181141, 0],
           [0, 1.5690706849093181141, 0, 0, 0, 
            0, 1.5690706849093181141, 0, 1.5690706849093181141, 0, 
            0, 0, 0, 0, 0],
           [0.1997329217870320861, 0, 0, -1.1983975307221925721, 0, 
            0, 0, 0, 0, 0, 
            0.1997329217870320861, 0, 0, 0, 0],
           [0, 0, -0.56493001368725082045, 0, 0, 
            0, 0, 1.6947900410617524614, 0, 0, 
            0, 0, 0, 0, 0],
          [-0.15098389705165754515, 0, 0, 0, 0, 
            0.90590338230994538193, 0, 0, 0, 0, 
            0.15098389705165754515, 0, -0.90590338230994538193, 0, 0],
           [0, 0, 0.64057042473119185644, 0, 0, 
            0, 0, 0.64057042473119185644, 0, -0.85409389964158910491, 
            0, 0, 0, 0, 0],
           [0.10128307719460090397, 0, 0, 0.20256615438920180794, 0, 
           -0.81026461755680723176, 0, 0, 0, 0, 
            0.10128307719460090397, 0, -0.81026461755680723176, 0, 0.27008820585226911426],
           [0, 0, 0, 0, 0.64057042473119185644, 
            0, 0, 0, 0, 0, 
            0, 0.64057042473119185644, 0, -0.85409389964158910491, 0],
           [0, -0.3019677941033150903, 0, 0, 0, 
            0, -0.3019677941033150903, 0, 1.8118067646198907639, 0, 
            0, 0, 0, 0, 0],
           [0, 0, 0, 0, -1.6947900410617524614, 
            0, 0, 0, 0, 0, 
            0, 0.56493001368725082045, 0, 0, 0],
           [0, 0.79893168714812834441, 0, 0, 0, 
            0,-0.79893168714812834441, 0, 0, 0, 
            0, 0, 0, 0, 0]]
    
    m = np.asarray(m)
    
    n   = len(m)
    res = np.array([np.dot(m[i], ten) for i in range(n)])

    return res

def compute_bispectrum(ten):

    esh = tijk_3d_sym_to_esh(ten)
    bs = np.zeros((5, 5, 5))
    
    # Coeff 0 0 0: */
    bs[0, 0, 0] = esh[0]**3
    
    # Coeff 0 2 2: */
    bs[0, 2, 2] = esh[0] * (esh[1]**2 + esh[2]**2 + esh[3]**2 + esh[4]**2 + esh[5]**2)
    
    # Coeff 0 4 4: */
    bs[0, 4, 4] = (esh[0] * (esh[10]**2 + esh[11]**2 + esh[12]**2 + esh[13]**2 + esh[14]**2 + 
                             esh[6]**2  + esh[7]**2  + esh[8]**2  + esh[9]**2) )
                      
    # Coeff 2 2 2: */
    bs[2, 2, 2] = (1.603567451474546379  * esh[1]**2 * esh[3] -
                   0.8017837257372731895 * esh[2]**2 * esh[3] -
                   0.5345224838248487931 * esh[3]**2 * esh[3] -
                   0.8017837257372731895 * esh[4]**2 * esh[3] +
                   1.603567451474546379  * esh[5]**2 * esh[3] -
                   2.777460299317654142  * esh[2] * esh[4] * esh[5] +
                   esh[1] * (-1.388730149658827071*esh[2]**2 + 1.388730149658827071*esh[4]**2) )
    
    # Coeff 2 2 4: */
    bs[2, 2, 4] = (1.069044967649697455  * esh[12] * esh[2] * esh[4] +
                   1.309307341415954173  * esh[11] * esh[3] * esh[4] -
                   0.3779644730092272439 * esh[11] * esh[2] * esh[5] +
                   0.925820099772551419  * esh[12] * esh[3] * esh[5] + 
                   esh[13] * esh[2] * esh[5] +
                   esh[10] * (-0.4780914437337574485 * esh[2]**2 +
                               0.717137165600636228  * esh[3]**2 -
                               0.4780914437337574485 * esh[4]**2 +
                               0.1195228609334393621 * esh[5]**2) +
                   esh[1]**2 * (0.1195228609334393621 * esh[10] +
                                0.707106781186547524  * esh[6]) -
                   0.707106781186547524  * esh[5] * esh[5] * esh[6] -
                   0.5345224838248487275 * esh[2] * esh[2] * esh[8] -
                   0.5345224838248487275 * esh[4] * esh[4] * esh[8] +
                   1.309307341415954173  * esh[2] * esh[3] * esh[9] -
                   0.3779644730092272439 * esh[4] * esh[5] * esh[9] +
                   esh[4] * esh[5] * esh[7] +
                   esh[1] * (0.3779644730092272439 * esh[11] * esh[4] +
                             1.414213562373095049  * esh[14] * esh[5] +
                             0.925820099772551419  * esh[3]  * esh[8] -
                             0.3779644730092272439 * esh[2]  * esh[9] + 
                             esh[13] * esh[4] + esh[2] * esh[7]) )
                       
    # Coeff 2 4 4: */
    bs[2, 4, 4]=(-0.5617690055391233399 * esh[11] * esh[12] * esh[2] -
                  0.825722823844770433  * esh[12] * esh[13] * esh[2] -
                  0.873862897505303026  * esh[13] * esh[14] * esh[2] -
                  0.5096471914376254908 * esh[10] * esh[10] * esh[3] -
                  0.4332001127219817227 * esh[11] * esh[11] * esh[3] -
                  0.2038588765750502241 * esh[12] * esh[12] * esh[3] +
                  0.1783765170031689218 * esh[13] * esh[13] * esh[3] +
                  0.713506068012675687  * esh[14] * esh[14] * esh[3] +
                  0.873862897505303026  * esh[13] * esh[4]  * esh[6] -
                  0.4670993664969137761 * esh[12] * esh[5]  * esh[6] +
                  0.713506068012675687  * esh[3]  * esh[6]  * esh[6] +
                  0.825722823844770433  * esh[12] * esh[4]  * esh[7] -
                  0.873862897505303026  * esh[14] * esh[4]  * esh[7] -
                  0.700649049745370625  * esh[11] * esh[5]  * esh[7] -
                  0.873862897505303026  * esh[2]  * esh[6]  * esh[7] +
                  0.1783765170031689218 * esh[3]  * esh[7]  * esh[7] +
                  0.5617690055391233399 * esh[11] * esh[4]  * esh[8] -
                  0.825722823844770433  * esh[13] * esh[4]  * esh[8] +
                  0.4670993664969137761 * esh[14] * esh[5]  * esh[8] -
                  0.825722823844770433  * esh[2]  * esh[7]  * esh[8] -
                  0.2038588765750502241 * esh[3]  * esh[8]  * esh[8] -
                  0.5617690055391233399 * esh[12] * esh[4]  * esh[9] -
                  0.88273482950474958   * esh[11] * esh[5]  * esh[9] +
                  0.700649049745370625  * esh[13] * esh[5]  * esh[9] -
                  0.5617690055391233399 * esh[2]  * esh[8]  * esh[9] -
                  0.4332001127219817227 * esh[3]  * esh[9]  * esh[9] +
                  esh[10] * (-0.2791452631195412426 * esh[11] * esh[4] +
                              1.184313050927584099  * esh[12] * esh[5] -
                              0.2791452631195412426 * esh[2]  * esh[9]) +
                  esh[1] * (0.44136741475237479   * esh[11] * esh[11] +
                            0.700649049745370625  * esh[11] * esh[13] +
                            0.4670993664969137761 * esh[12] * esh[14] +
                            1.184313050927584099  * esh[10] * esh[8] +
                            0.4670993664969137761 * esh[6]  * esh[8] +
                            0.700649049745370625  * esh[7]  * esh[9] -
                            0.44136741475237479   * esh[9]  * esh[9]) )
    # Coeff 4 4 4: */
    #print esh
    bs[4, 4, 4] = (0.4022911406409067636 * esh[10] * esh[10] * esh[10] -
                   1.189993242032876571  * esh[12] * esh[12] * esh[6] +
                   2.098950786844323825  * esh[11] * esh[13] * esh[6] -
                   2.098950786844323825  * esh[11] * esh[14] * esh[7] -
                   0.8995503372189958914 * esh[11] * esh[11] * esh[8] +
                   0.7933288280219177136 * esh[11] * esh[13] * esh[8] +
                   1.189993242032876571  * esh[6]  * esh[8]  * esh[8] -
                   2.098950786844323825  * esh[13] * esh[14] * esh[9] -
                   2.098950786844323825  * esh[6]  * esh[7]  * esh[9] +
                   0.7933288280219177136 * esh[7]  * esh[8]  * esh[9] +
                   0.8995503372189958914 * esh[8]  * esh[9]  * esh[9] +
                   esh[12] * (-0.7933288280219177136 * esh[11] * esh[7] +
                              2.379986484065753141   * esh[14] * esh[8] +
                              1.799100674437991783   * esh[11] * esh[9] +
                              0.7933288280219177136  * esh[13] * esh[9]) +
                   esh[10] * (0.6034367109613601454 * esh[11]**2 -
                              0.7375337578416624186 * esh[12]**2 -
                              1.408018992243173617  * esh[13]**2 +
                              0.938679328162115745  * esh[14]**2 +
                              0.938679328162115745  * esh[6]**2  -
                              1.408018992243173617  * esh[7]**2  -
                              0.7375337578416624186 * esh[8]**2  +
                              0.6034367109613601454 * esh[9]**2) )
    #print bs[4, 4, 4]                    
    signs = np.sign(bs)
    bs = signs * np.absolute(bs)**(1./3.)
    return bs

def load_transform(file_path, l):

    x, options = nrrd.read(file_path)
    data = x.transpose()

    m =[
        0.28209479177387819515, 0.54627421529603970018, 0, 
        -0.31539156525252004526, 0, 0, 0.62583573544917614484, 0, 
        -0.47308734787878004013, 0, 0.31735664074561298342, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0.27313710764801985009, 0, 0, 0, 0, 0, 0, 
        -0.23654367393939002007, 0, 0.62583573544917614484,
        0, 0, -0.27313710764801979458, 0, 0, 0, 0, -0.44253269244498261159, 
        0, 0.50178490766796690625, 0, 0, 0, 0, 0,
        0.094031597257959328995, -2.6239092990136782376e-17, 0, 
        -0.10513052175084000583, 0, 0, -0.62583573544917614484, 0, 
        -1.1695830935687379082e-17, 0, 0.10578554691520429543, 0, 0, 0, 0,
        0, 0, 0, 0, -0.091045702549339968535, 0, 0, 0, 0, 0, 0, 
        0.16726163588932230208, 0, -0.44253269244498261159, 0,
        0.094031597257959398384, 0.091045702549339926901, 0, 
        0.052565260875419995978, 0, 0, 9.2621981986301462932e-18, 0, 
        0.47308734787878004013, 0, -0.42314218766081729273, 0, 0, 0, 0,
        -0, -0, -0, -0, -0, 0.27313710764801979458, -0, -0, 0, 0, 0, 0, 
        -0.23654367393939002007, 0, -0.62583573544917614484,
        0, 0, -0.091045702549339954657, 0, 0, 0, 0, 0.44253269244498261159, 
        0, 0.16726163588932227433, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0.091045702549339926901, 0, 0, 0, 0, 0, 0, 
        0.47308734787878004013, 0, 0,
        0, 0, -0.27313710764801979458, 0, 0, 0, 0, 0, 0, 
        -0.66904654355728920834, 0, 0, 0, 0, 0,
        0.28209479177387819515, -0.54627421529603958916, 0, 
        -0.31539156525252004526, 0, 0, 0.62583573544917614484, 0, 
        0.47308734787878004013, 0, 0.31735664074561298342, 0, 0, 0, 0,
        0, 0, 0, 0, -0.27313710764801979458, 0, 0, 0, 0, 0, 0, 
        0.50178490766796690625, 0, 0.44253269244498261159, 0,
        0.094031597257959370628, -0.091045702549339926901, 0, 
        0.052565260875420016795, 0, 0, 9.262198198630141671e-18, 0, 
        -0.47308734787878004013, 0, -0.42314218766081729273, 0, 0, 0, 0,
        -0, -0, -0, -0, -0.27313710764801979458, -0, -0, -0, 0, 0, 0, 
        -0.66904654355728920834, 0, 0, 0,
        0.28209479177387819515, 4.5022398477899912361e-18, -0, 
        0.63078313050504009052, -0, 0, 2.7789509525674121332e-17, -0, 
        2.3394324492700281759e-17, 0, 0.84628437532163458545, 0, 0, 0, 0
    ]

    m = np.array(m).reshape((15, 15))

    ten = np.dot(data, m)



    bs = np.asarray([compute_bispectrum(row) for row in ten])


    matrix = []
    for i in xrange(len(bs)):
        row = []
        for j in xrange(len(bs[i])):
            for k in xrange(len(bs[i][j])):
                for m in xrange(len(bs[i][j][k])):

                    row.append(bs[i][j][k][m])

        matrix.append(row)

    # add the label class
    
    labels = [l] * len(matrix)
    
    return matrix, labels


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
        "n_inputs"              : 125,       # Number of network input signals
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
df_res.to_excel("sh_bs.xlsx")
df_res

'''



