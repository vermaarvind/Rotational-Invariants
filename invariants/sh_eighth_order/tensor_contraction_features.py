"""
Tensor Contraction features 

"""
import cmath
import math
import pandas
import numpy as np
import scipy.io
from matplotlib import pyplot as pot
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sympy.physics.quantum.cg import CG
from sympy import S
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from activation_functions import sigmoid_function, tanh_function, linear_function, LReLU_function, ReLU_function, elliot_function, symmetric_elliot_function, softmax_function
from cost_functions import sum_squared_error
from learning_algorithms import backpropagation, scaled_conjugate_gradient, scipyoptimize, resilient_backpropagation
from neuralnet import NeuralNet
from tools import Instance
import datetime
import time
from sklearn.metrics import confusion_matrix
import openpyxl

###########_________loading .mat file_________########################

mat = scipy.io.loadmat('ODFcoef_SNR10.mat')
x = mat['x']
sizes = np.shape(np.array(mat['x']))

########_______generating the invariants_____#####################

def Conv_to_comp(array):
    Cx=[0 for i in range(len(array))]
    i=1
    c = []
    for i in range(0,9):
        for j in range(-2*i,2*i+1):
            c.append(j)
    for k in range(0,len(array)):
        if (c[k]==0):
            Cx[k]=array[k]
        elif (c[k]>0):
            Cx[k]=1/math.sqrt(2)*(array[k]+cmath.sqrt(-1)*array[k-2*c[k]])
        else:
            Cx[k]=1/math.sqrt(2)*math.pow((-1),c[k])*(array[k+2*abs(c[k])]-cmath.sqrt(-1)*array[k])

    return Cx

def CalcTensor(Cx, l, l1, l2):
    Ten = [[0 for x in range(1)] for x in range(2*l+1)]
    result = 0
        #if (l>(l1+l2) or l<abs(l1-l2)):
        # print('Invalid tensor')
        # Ten = -1
    if(l1==0):
        s1 = 1
    elif(l1==2):
        s1 = 4
    elif(l1==4):
        s1 = 11
    elif(l1==6):
        s1 = 22
    elif(l1==8):
        s1 = 37
    
    if(l2==0):
        s2 = 1
    elif(l2==2):
        s2 = 4
    elif(l2==4):
        s2 = 11
    elif(l2==6):
        s2 = 22
    elif(l2==8):
        s2 = 37

    #column of length 2l+1
    jj = 0
    for m in range(-l,l+1):
        for m1 in range(-l1,l1+1):
            for m2 in range(-l2,l2+1):
                el=float(CG(l1,m1,l2,m2,l,m).doit())
                Ten[jj][0]=Ten[jj][0]+el*np.conjugate(Cx[m1+s1-1])*np.conjugate(Cx[m2+s2-1])
        jj = jj + 1
    return Ten

Mas1 = [[0 for i in xrange(5)] for i in xrange(300)]
for j in range(0, len(x)):
    Invar2 = [0 for i in range(5)]
    Cx1 = Conv_to_comp(x[j])
    Invar2[0]=Cx1[0]*CalcTensor(Cx1,0,2,2)[0][0]
    arr= np.array(CalcTensor(Cx1,2,2,2)).transpose()
    
    Invar2[1]=0
    count=0
    
    for i in range(1, 6):
        Invar2[1] = Invar2[1] + (np.conjugate(Cx1[i]))*np.conjugate(arr[0][count])
        count = count+1

    count=0
    Invar2[2]=0
    arr1= np.array(CalcTensor(Cx1,4,2,2)).transpose()
    for i in range(6, 15):
        Invar2[2] = Invar2[2] + (np.conjugate(Cx1[i]))*np.conjugate(arr1[0][count])
        count = count+1

    count=0
    Invar2[3]=0
    arr2= np.array(CalcTensor(Cx1,6,4,4)).transpose()
    for i in range(15, 28):
        Invar2[3] = Invar2[3] + (np.conjugate(Cx1[i]))*np.conjugate(arr2[0][count])
        count = count+1

    count=0
    Invar2[4]=0
    arr3= np.array(CalcTensor(Cx1,8,4,4)).transpose()
    for i in range(28, 45):
        Invar2[4] = Invar2[4] + (np.conjugate(Cx1[i]))*np.conjugate(arr3[0][count])
        count = count+1

    count=0
    Invar2[3]=0
    arr2= np.array(CalcTensor(Cx1,6,4,4)).transpose()
    for i in range(15, 28):
        Invar2[3] = Invar2[3] + (np.conjugate(Cx1[i]))*np.conjugate(arr2[0][count])
        count = count+1

    Mas1[j]=Invar2
Mas1=np.array(Mas1)


df = pandas.DataFrame(Mas1)


#################_____classification of fibers_______##########################

   
def RandomForest_cross_validation(df):
    # generate labels
    for i in xrange(5):
        df[i] = df[i].apply(lambda x: x.real)   
    
    # generate labels
    df ['y'] =  0

    for i in range(len(df)):
        if i > 200:
            df.loc[i, 'y'] = 2
        elif i > 100:
            df.loc[i, 'y'] = 1

    y = df["y"].values
    df = df.drop(["y"], 1)
    X = df.values
    
    
    # split randomly into train\test datasets
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
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
        scores =cross_val_score(dlf, Xtrain, ytrain, cv = 10)
        acc_avg = scores.mean()
        print "Score = ", acc_avg
        print "Number of trees =  ", n

                
        parameters.append(n)
        acc_all.append(acc_avg)
                
        if acc_avg > best_acc:
                    best_acc = acc_avg
                    best_par = n
                    
        if len(acc_all) >=3:
            if acc_avg < acc_all[-2] or acc_avg == acc_all[-3]:
                break
                
        time_list_RF.append([n, acc_avg, scores, time.time() - timeStart])
                
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print dlf
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"    
    
    print "The best CV Accuracy is ", best_acc
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

    #print "Train model"
    timeStart_train = time.time()
    dlf.fit(Xtrain, ytrain)
    train_score = dlf.score(Xtrain, ytrain)
    time_train = time.time() - timeStart_train

    #print "Predict"
    timeStart_test = time.time()
    pred = dlf.predict(Xtest)
    test_score = dlf.score(Xtest, ytest)
    time_test = time.time() - timeStart_test
        
    time_train_predict.append([n, time_train, train_score, time_test, test_score])      

    cm = confusion_matrix(ytest, pred)
   
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
    
    return time_train_predict, cm 
    
    
def KNeighbors(df):
    # generate labels
    for i in xrange(5):
        df[i] = df[i].apply(lambda x: x.real)   
    
    # generate labels
    df ['y'] =  0

    for i in range(len(df)):
        if i > 200:
            df.loc[i, 'y'] = 2
        elif i > 100:
            df.loc[i, 'y'] = 1

    y = df["y"].values
    df = df.drop(["y"], 1)
    X = df.values
    
    
    # split randomly into train\test datasets
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print "kNN Classifier"
    print "--------------------------------------------------------"
    print "Grid search and Cross validation"
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
        scores =cross_val_score(dlf, Xtrain, ytrain, cv = 10)
        acc_avg = scores.mean()
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
        time_list_KN.append([n, acc_avg, scores, time.time() - timeStart])
    
    
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print dlf
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"    

    print "The best CV Accuracy is ", best_acc
    print "The best minimum number of neighbors is ", best_par
    
    
    time_train_predict = []  
    
    
        
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print "Training and Prediction"
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"


    dlf = KNeighborsClassifier(n_neighbors=best_par, leaf_size=30, p=2, metric='minkowski', metric_params=None)

    #print "Train model"
    timeStart_train = time.time()
    dlf.fit(Xtrain, ytrain)
    train_score = dlf.score(Xtrain, ytrain)
    time_train = time.time() - timeStart_train

    #print "Predict"
    timeStart_test = time.time()
    pred = dlf.predict(Xtest)
    test_score = dlf.score(Xtest, ytest)
    time_test = time.time() - timeStart_test
        
        
    time_train_predict.append([n, time_train, train_score, time_test, test_score])      

    cm = confusion_matrix(ytest, pred)
   
    
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
    
    return time_train_predict, cm
    
'''    
def FCNN(df):
    
    # generate labels
    for i in xrange(5):
        df[i] = df[i].apply(lambda x: x.real)   
    
    # generate labels
    df ['y'] =  0

    for i in range(len(df)):
        if i > 200:
            df.loc[i, 'y'] = 2
        elif i > 100:
            df.loc[i, 'y'] = 1

    y = df["y"].values
    df = df.drop(["y"], 1)
    X = df.values

    training_one = []
    for i in xrange(len(X)):
        training_one.append(Instance(X[i], [y[i]]))

        
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print "Defining Neural Network"
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    
    settings = {
        # Required settings
        "cost_function"         : sum_squared_error,
        "n_inputs"              : 5,       # Number of network input signals
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
RandomForest_cross_validation(df)

KNeighbors(df)

'''

df_res = pandas.DataFrame()
for i in xrange(len(time_list_RF)):
    df_res.loc["CV Time (sec)", "RF ({} trees)".format(time_list_RF[i][0])] = time_list_RF[i][3]
    df_res.loc["CV Accuracy", "RF ({} trees)".format(time_list_RF[i][0])] = time_list_RF[i][1]

df_res.loc["Training Time (sec)", "RF ({} trees)".format(time_train_predict_RF[0][0])] = time_train_predict_RF[0][1]
df_res.loc["Training Accuracy", "RF ({} trees)".format(time_train_predict_RF[0][0])] = time_train_predict_RF[0][2]
df_res.loc["Prediction Time (sec)", "RF ({} trees)".format(time_train_predict_RF[0][0])] = time_train_predict_RF[0][3]
df_res.loc["Prediction Accuracy", "RF ({} trees)".format(time_train_predict_RF[0][0])] = time_train_predict_RF[0][4]

df_res.loc["Confusion matrix", "RF ({} trees)".format(time_train_predict_RF[0][0])] = str(cm_RF)

for i in xrange(len(time_list_KN)):
    df_res.loc["CV Time (sec)", "KN ({} neighbors)".format(time_list_KN[i][0])] = time_list_KN[i][3]
    df_res.loc["CV Accuracy", "KN ({} neighbors)".format(time_list_KN[i][0])] = time_list_KN[i][1]

df_res.loc["Training Time (sec)", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = time_train_predict_KN[0][1]
df_res.loc["Training Accuracy", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = time_train_predict_KN[0][2]
df_res.loc["Prediction Time (sec)", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = time_train_predict_KN[0][3]
df_res.loc["Prediction Accuracy", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = time_train_predict_KN[0][4]

df_res.loc["Confusion matrix", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = str(cm_KN)


df_res = df_res.fillna('')
df_res.to_excel("tensor_contraction_features.xlsx")
df_res

'''
#FCNN(df)




