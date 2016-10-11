"""
SH coeff. with training data augmented with rotated versions

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
from sympy.physics.quantum.cg import CG
from sympy import S
import pyshtools as shtools
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
import time
from sklearn.metrics import confusion_matrix
import openpyxl
from activation_functions import sigmoid_function, tanh_function, linear_function, LReLU_function, ReLU_function, elliot_function, symmetric_elliot_function, softmax_function
from cost_functions import sum_squared_error
from learning_algorithms import backpropagation, scaled_conjugate_gradient, scipyoptimize, resilient_backpropagation
from neuralnet import NeuralNet
from tools import Instance

##################################################################

mat = scipy.io.loadmat('ODFcoef_SNR20.mat')

x = mat['x']
sizes = np.shape(np.array(mat['x']))
df1 = pandas.DataFrame(x)

##################################################################
#-----This function gives rotated SH coeff. coming from Cx-------#
def SHRotations(coeff):

    #---- input parameters ----#
    lmax = 8     #order of the SH
    alpha, beta, gamma = 159., 80., 145. #angles for rotation

    #--derived parameters --#
    angles = np.radians([alpha, beta, gamma])
    dj_matrix = shtools.djpi2(lmax)

    return shtools.SHRotateRealCoef(coeff, angles, dj_matrix)

################################################################################
#-----This function converts the values into complex and returns Cx------#
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

def Conv_to_real(array):
    Cx=[0 for i in range(len(array))]
    i=1
    c = []
    for i in range(0,9):
        for j in range(-2*i,2*i+1):
            c.append(j)
    for k in range(0,len(array)):
        if (c[k]==0):
            Cx[k]=array[k].real
        elif (c[k]>0):
            Cx[k]=(array[k].real)*math.sqrt(2)
        else:
            Cx[k]=-math.pow((-1),c[k])*(array[k].imag)*math.sqrt(2)

    return Cx

def BackToComplex(array_real_imag):
    #print array_real_imag[0]  #real
    #print array_real_imag[1]  #imag
    initial_form=range(45)
    k=0
    for i in range(0,9):
        for j in range(0,i+1):
            initial_form[k]=complex(array_real_imag[0][i][j],-array_real_imag[1][i][j])
            if (j > 0):
                initial_form[k]/=math.sqrt(2)     # Check for sign difference
            k=k+1

    in_f=range(45)
    for i in range(0,len(in_f)):
        in_f[i]=initial_form[i]
    
    in_f[1]=np.conj(initial_form[5])
    in_f[2]=-np.conj(initial_form[4])

    in_f[6]=np.conj(initial_form[14])
    in_f[7]=-np.conj(initial_form[13])
    in_f[8]=np.conj(initial_form[12])
    in_f[9]=-np.conj(initial_form[11])

    in_f[15]=np.conj(initial_form[27])
    in_f[16]=-np.conj(initial_form[26])
    in_f[17]=np.conj(initial_form[25])
    in_f[18]=-np.conj(initial_form[24])
    in_f[19]=np.conj(initial_form[23])
    in_f[20]=-np.conj(initial_form[22])

    in_f[28]=np.conj(initial_form[44])
    in_f[29]=-np.conj(initial_form[43])
    in_f[30]=np.conj(initial_form[42])
    in_f[31]=-np.conj(initial_form[41])
    in_f[32]=np.conj(initial_form[40])
    in_f[33]=-np.conj(initial_form[39])
    in_f[34]=np.conj(initial_form[38])
    in_f[35]=-np.conj(initial_form[37])

    return in_f

def Creating_Matrix(a):
    '''
    e.g : 

    1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 
    4 5 6 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 
    11 12 13 14 15...
    .................
    ...............45
    '''
    mat=[[0 for x in range(9)] for x in range(9)]

    k=0
    for i in range(0,9):
        for j in range(0,i+1):
            if (i % 2) == 0:   # Michael 
                mat[i][j]=a[k]
                
                if (j > 0):
                    mat[i][j]*=math.sqrt(2)
            k=k+1

    return mat

def Rotation(Cx0): 

    #---Rotation of Cx---#
    # Cx is an array of 45 elements, each of them is in form of (a+ib)
    RealPart=[]
    for i in range(0,len(Cx0)):
        RealPart.append(Cx0[i].real)

    rp=Creating_Matrix(RealPart)


    ImagPart=[]
    for i in range(0,len(Cx0)):
        ImagPart.append(-Cx0[i].imag)

    ip=Creating_Matrix(ImagPart)

    SH_coeff=[[rp[0],rp[1],rp[2],rp[3],rp[4],rp[5],rp[6],rp[7],rp[8]],[ip[0],ip[1],ip[2],ip[3],ip[4],ip[5],ip[6],ip[7],ip[8]]]

    rotated_array=SHRotations(SH_coeff)
    final_rotated=BackToComplex(rotated_array)

    return final_rotated 

x_real = []
for j in range(0, len(x)):
    Cx = Conv_to_comp(x[j])
    rotated_Cx=Rotation(Cx)                    
    Cx_real = Conv_to_real(rotated_Cx)         
    Cx_real=np.array(Cx_real)    
    Cx_real = Conv_to_real(rotated_Cx)
    x_real.append(Cx_real)

df2 = pandas.DataFrame(x_real)

df = pandas.concat([df1, df2])
df.index = range(len(df))

#print "Data set size is ", df.shape

def RandomForest_cross_validation(df):
    startTime = time.time()  
    # generate labels
    df ['y'] =  0

    for i in range(len(df)):
        if i > 500:
            df.loc[i, 'y'] = 2
        elif i > 400:
            df.loc[i, 'y'] = 1
        elif i > 300:
            df.loc[i, 'y'] = 0
        elif i > 200:
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

    #train
    timeStart_train = time.time()
    dlf.fit(Xtrain, ytrain)
    train_score = dlf.score(Xtest, ytest)
    time_train = time.time() - timeStart_train

    #Predict
    timeStart_test = time.time()
    pred = dlf.predict(Xtest)
    test_score = dlf.score(Xtest, ytest)
    time_test = time.time() - timeStart_test
        
    time_train_predict.append([n, time.time() - timeStart_train, train_score, time_test, test_score])      

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
    
    return time_list_RF, time_train_predict, cm


def KNeighbors(df):  
    # generate labels
    df ['y'] =  0

    for i in range(len(df)):
        if i > 500:
            df.loc[i, 'y'] = 2
        elif i > 400:
            df.loc[i, 'y'] = 1
        elif i > 300:
            df.loc[i, 'y'] = 0
        elif i > 200:
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
    print "Grid search"
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    #Random forest
    # grid search by the best parameters
    parameters = []
    acc_all = []
    time_list_KN = []
    best_acc = 0
    best_par = 0
    for n in [5, 10, 20, 30, 50]:
        
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

    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print dlf
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"    

    print "The best CV Accuracy is ", best_acc
    print "The best minimum number of neighbors is ", best_par


    print "--------------------------------------------------------"
    print "--------------------------------------------------------"



    time_train_predict = []  
    
    
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print "Training and Prediction"
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"


    dlf = KNeighborsClassifier(n_neighbors=best_par, leaf_size=30, p=2, metric='minkowski', metric_params=None)

    #train
    timeStart_train = time.time()
    dlf.fit(Xtrain, ytrain)
    train_score = dlf.score(Xtest, ytest)
    time_train = time.time() - timeStart_train

    #Predict
    timeStart_test = time.time()
    pred = dlf.predict(Xtest)
    test_score = dlf.score(Xtest, ytest)
    time_test = time.time() - timeStart_test
        
    time_train_predict.append([n, time.time() - timeStart_train, train_score, time_test, test_score])      

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
    
    return time_list_KN, time_train_predict, cm 

'''
def FCNN(df):
    
    # generate labels
    df ['y'] =  0
    
    for i in range(len(df)):
        if i > 500:
            df.loc[i, 'y'] = 2
        elif i > 400:
            df.loc[i, 'y'] = 1
        elif i > 300:
            df.loc[i, 'y'] = 0
        elif i > 200:
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
        "n_inputs"              : 45,       # Number of network input signals
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
time_list_RF, time_train_predict_RF, cm_RF  = RandomForest_cross_validation(df)


time_list_KN, time_train_predict_KN, cm_KN = KNeighbors(df)
'''
# In[5]:
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
df_res.to_excel("augmented_sh_coeff.xlsx")
df_res

'''
#FCNN(df)





