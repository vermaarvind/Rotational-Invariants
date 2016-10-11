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
import pyshtools as shtools
import datetime
import time

####################___Loading the train ODF .nrrd file____##########################

train_data_1, options = nrrd.read('one-train-odfs.nrrd')
train_data_2, options = nrrd.read('two-train-odfs.nrrd')
train_data_3, options = nrrd.read('three-train-odfs.nrrd')

train1 = train_data_1.transpose()
train2 = train_data_2.transpose()
train3 = train_data_3.transpose()

train_temp = np.concatenate((train1, train2), axis=0)
x = np.concatenate((train_temp, train3), axis=0)  # use this wihout pandas 

df1 = pandas.DataFrame(x)

print "pure non rotated training data : ", df1.shape

####################___Loading the test ODF .nrrd file____##########################

test_data_1, options = nrrd.read('one-test-odfs.nrrd')
test_data_2, options = nrrd.read('two-test-odfs.nrrd')
test_data_3, options = nrrd.read('three-test-odfs.nrrd')

test1 = test_data_1.transpose()
test2 = test_data_2.transpose()
test3 = test_data_3.transpose()

test_temp = np.concatenate((test1, test2), axis=0)
y = np.concatenate((test_temp, test3), axis=0)

df3 = pandas.DataFrame(y)

print "test data : ", df3.shape


#############################################################################

#############################################################################
##                                                                         ##
##     Rotating the training data and appending the training dataset       ##
##                                                                         ##
#############################################################################

#############################################################################
#-----This function gives rotated SH coeff. coming from Cx-------#

def SHRotations(coeff):

    #---- input parameters ----#

    lmax = 4     #order of the SH
    alpha, beta, gamma = 165., 175., 100. #angles for rotation

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
    for i in range(0,5):
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
    for i in range(0,5):
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
    initial_form=range(15)
    k=0
    for i in range(0,5):
        for j in range(0,i+1):
            initial_form[k]=complex(array_real_imag[0][i][j],-array_real_imag[1][i][j])
            if (j > 0):
                initial_form[k]/=math.sqrt(2)  # Check for sign difference
            k=k+1

    in_f=range(15)
    for i in range(0,len(in_f)):
        in_f[i]=initial_form[i]
    
    in_f[1]=np.conj(initial_form[5])
    in_f[2]=-np.conj(initial_form[4])

    in_f[6]=np.conj(initial_form[14])
    in_f[7]=-np.conj(initial_form[13])
    in_f[8]=np.conj(initial_form[12])
    in_f[9]=-np.conj(initial_form[11])

    return in_f

def Creating_Matrix(a):
    '''
    e.g : 

    1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 
    4 5 6 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 
    11 12 13 14 15...

    '''
    mat=[[0 for x in range(5)] for x in range(5)]

    k=0
    for i in range(0,5):
        for j in range(0,i+1):
            if (i % 2) == 0:   # Michael 
                mat[i][j]=a[k]
                
                if (j > 0):
                    mat[i][j]*=math.sqrt(2)
            k=k+1

    return mat

def Rotation(Cx0): 

    #---Rotation of Cx---#

    # Cx is an array of 15 elements, each of them is in form of (a+ib)
    RealPart=[]
    for i in range(0,len(Cx0)):
        RealPart.append(Cx0[i].real)

    rp=Creating_Matrix(RealPart)


    ImagPart=[]
    for i in range(0,len(Cx0)):
        ImagPart.append(-Cx0[i].imag)  #Sign covention for the SH library convention sign diffrence 

    ip=Creating_Matrix(ImagPart)
    
    #make it to the form of input which we can put into SHRotation
    SH_coeff=[[rp[0],rp[1],rp[2],rp[3],rp[4]],[ip[0],ip[1],ip[2],ip[3],ip[4]]]

    rotated_array=SHRotations(SH_coeff)
    final_rotated=BackToComplex(rotated_array)

    return final_rotated 

x_real = []
for j in range(0, len(x)): #loop is running 300 times
    Cx = Conv_to_comp(x[j])
    rotated_Cx=Rotation(Cx)                    
    Cx_real = Conv_to_real(rotated_Cx)         
    Cx_real=np.array(Cx_real)    
    Cx_real = Conv_to_real(rotated_Cx)
    x_real.append(Cx_real)

df_trimmed = []
multiplier=0
innercounter=1

while(innercounter+(multiplier*1000)-1<len(x_real)):
    if (innercounter<=500):
        #print innercounter+(multiplier*1000)-1
        df_trimmed.append(x_real[innercounter+(multiplier*1000)-1])        
    else:
        multiplier=multiplier+1
        innercounter=0
    innercounter = innercounter+1

trimmed = np.array(df_trimmed)

#dftrimmed=pandas.DataFrame(df_trimmed)

df_slice=pandas.DataFrame(trimmed)

#print df_slice.shape

df_train = pandas.concat([df1, df_slice])
df_train.index = range(len(df_train))

#print df_train.shape

print "final training data set appended with only half respective rotated versions : ", df_train.shape

#quit()

#-----This function gives rotated SH coeff. coming from Cx-------#

def SHRotations(coeff):

    #---- input parameters ----#

    lmax = 4     #order of the SH
    alpha, beta, gamma = 165., 175., 100. #angles for rotation

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
    for i in range(0,5):
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
    for i in range(0,5):
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
    initial_form=range(15)
    k=0
    for i in range(0,5):
        for j in range(0,i+1):
            initial_form[k]=complex(array_real_imag[0][i][j],-array_real_imag[1][i][j])
            if (j > 0):
                initial_form[k]/=math.sqrt(2)  # Check for sign difference
            k=k+1

    in_f=range(15)
    for i in range(0,len(in_f)):
        in_f[i]=initial_form[i]
    
    in_f[1]=np.conj(initial_form[5])
    in_f[2]=-np.conj(initial_form[4])

    in_f[6]=np.conj(initial_form[14])
    in_f[7]=-np.conj(initial_form[13])
    in_f[8]=np.conj(initial_form[12])
    in_f[9]=-np.conj(initial_form[11])

    return in_f

def Creating_Matrix(a):
    '''
    e.g : 

    1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 
    4 5 6 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 
    11 12 13 14 15...

    '''
    mat=[[0 for y in range(5)] for y in range(5)]

    k=0
    for i in range(0,5):
        for j in range(0,i+1):
            if (i % 2) == 0:   # Michael 
                mat[i][j]=a[k]
                
                if (j > 0):
                    mat[i][j]*=math.sqrt(2)
            k=k+1

    return mat

def Rotation(Cx0): 

    #---Rotation of Cx---#

    # Cx is an array of 15 elements, each of them is in form of (a+ib)
    RealPart=[]
    for i in range(0,len(Cx0)):
        RealPart.append(Cx0[i].real)

    rp=Creating_Matrix(RealPart)


    ImagPart=[]
    for i in range(0,len(Cx0)):
        ImagPart.append(-Cx0[i].imag)  #Sign covention for the SH library convention sign diffrence 

    ip=Creating_Matrix(ImagPart)
    
    #make it to the form of input which we can put into SHRotation
    SH_coeff=[[rp[0],rp[1],rp[2],rp[3],rp[4]],[ip[0],ip[1],ip[2],ip[3],ip[4]]]

    rotated_array=SHRotations(SH_coeff)
    final_rotated=BackToComplex(rotated_array)

    return final_rotated 

y_real = []
for j in range(0, len(y)): #loop is running 3000 times
    Cx = Conv_to_comp(x[j])
    rotated_Cx=Rotation(Cx)                    
    Cx_real = Conv_to_real(rotated_Cx)         
    Cx_real=np.array(Cx_real)    
    Cx_real = Conv_to_real(rotated_Cx)
    y_real.append(Cx_real)

df4 = pandas.DataFrame(y_real)

df_test = pandas.concat([df3, df4])
df_test.index = range(len(df_test))

#print " test data set (df_test) appended with respective rotated versions ", df_test.shape

##############################################################################################


df_test['y'] =  0

for i in range(len(df_test)):
        if i > 2500:
            df_test.loc[i, 'y'] = 2
        elif i > 2000:
            df_test.loc[i, 'y'] = 1
        elif i > 1500:
            df_test.loc[i, 'y'] = 0
        elif i > 1000:
            df_test.loc[i, 'y'] = 2
        elif i > 500:
            df_test.loc[i, 'y'] = 1
        else:
            df_test.loc[i, 'y'] = 0



'''
4001 - 4500 2
3501 - 4000 1
3001 - 3750 0
2001 - 3000 2
1001 - 2000 1
0    - 1000 0
    
'''

df_train['y'] =  0

for i in range(len(df_train)):
        if i > 4000:
            df_train.loc[i, 'y'] = 2
        elif i > 3500:
            df_train.loc[i, 'y'] = 1
        elif i > 3000:
            df_train.loc[i, 'y'] = 0
        elif i > 2000:
            df_train.loc[i, 'y'] = 2
        elif i > 1000:
            df_train.loc[i, 'y'] = 1
        else:
            df_train.loc[i, 'y'] = 0


df3['y'] =  0

for i in range(len(df3)):
        if i > 1000:
            df3.loc[i, 'y'] = 2
        elif i > 500:
            df3.loc[i, 'y'] = 1
        else:
            df3.loc[i, 'y'] = 0


#####################################################

y_train = df_train['y'].values
df_train = df_train.drop(['y'], 1)
X_train = df_train.values


y_test = df_test['y'].values
df_test = df_test.drop(['y'], 1)
X_test = df_test.values


y_test_nr = df3['y'].values
df3 = df3.drop(['y'], 1)
X_test_nr = df3.values


def RandomForest_cross_validation(X_train, y_train, X_test, y_test, y_test_nr, X_test_nr):
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
    
    print "The best Accuracy is ", best_acc
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
    train_score = dlf.score(X_test_nr, y_test_nr)
    time_train = time.time() - timeStart_train

    #Predict
    timeStart_test = time.time()
    pred = dlf.predict(X_test)
    test_score = dlf.score(X_test, y_test)
    time_test = time.time() - timeStart_test



    #Predict nr
    timeStart_test_nr = time.time()
    pred_nr = dlf.predict(X_test_nr)
    test_score_nr = dlf.score(X_test_nr, y_test_nr)
    time_test_nr = time.time() - timeStart_test_nr
        
    time_train_predict.append([n, time.time() - timeStart_train, train_score, time_test, test_score, time_test_nr, test_score_nr])      

    cm = confusion_matrix(y_test, pred)
    cm_nr = confusion_matrix(y_test_nr, pred_nr)
    
    #print "Training score for the rotated data : ", train_score
    print "Training time : ", time_train 
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"    
    #print "Test score : ", test_score
    #print "Test time : ", time_test
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    #print "Confusion matrix : "
    #print cm
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print "Test score nr : ", test_score_nr
    print "Test time nr : ", time_test_nr
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print "Confusion matrix nr : "
    print cm_nr
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    
    return time_list_RF, time_train_predict, cm, cm_nr
    
    
def KNeighbors(X_train, y_train, X_test, y_test, y_test_nr, X_test_nr):
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

    print "The best Accuracy is ", best_acc
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
        
    #Predict nr
    timeStart_test_nr = time.time()
    pred_nr = dlf.predict(X_test_nr)
    test_score_nr = dlf.score(X_test_nr, y_test_nr)
    time_test_nr = time.time() - timeStart_test_nr
        
    time_train_predict.append([n, time.time() - timeStart_train, train_score, time_test, test_score, time_test_nr, test_score_nr])      

    cm = confusion_matrix(y_test, pred)
    cm_nr = confusion_matrix(y_test_nr, pred_nr)
    
    #print "Training score : ", train_score
    print "Training time :", time_train 
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"    
    #print "Test score : ", test_score
    #print "Test time : ", time_test
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    #print "Confusion matrix : "
    #print cm
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print "Test score nr: ", test_score_nr
    print "Test time nr : ", time_test_nr
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    print "Confusion matrix nr: "
    print cm_nr
    print "--------------------------------------------------------"
    print "--------------------------------------------------------"
    
    return time_list_RF, time_train_predict, cm, cm_nr 

time_list_RF, time_train_predict_RF, cm_RF, cm_nr_RF  = RandomForest_cross_validation(X_train, y_train, X_test, y_test, y_test_nr, X_test_nr)


time_list_KN, time_train_predict_KN, cm_KN, cm_nr_KN = KNeighbors(X_train, y_train, X_test, y_test, y_test_nr, X_test_nr)
'''
def FCNN(X_train, y_train, X_test, y_test, y_test_nr, X_test_nr):
    
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

#FCNN(X_train, y_train, X_test, y_test)

#Saving the results in a excel file
df_res = pandas.DataFrame()


df_res.loc["Training Time (sec)", "RF ({} trees)".format(time_train_predict_RF[0][0])] = time_train_predict_RF[0][1]
df_res.loc["Training Accuracy", "RF ({} trees)".format(time_train_predict_RF[0][0])] = time_train_predict_RF[0][2]
df_res.loc["Prediction Time (sec)", "RF ({} trees)".format(time_train_predict_RF[0][0])] = time_train_predict_RF[0][3]
df_res.loc["Prediction Accuracy", "RF ({} trees)".format(time_train_predict_RF[0][0])] = time_train_predict_RF[0][4]

df_res.loc["Confusion matrix", "RF ({} trees)".format(time_train_predict_RF[0][0])] = str(cm_RF)
df_res.loc["Prediction Time nr (sec)", "RF ({} trees)".format(time_train_predict_RF[0][0])] = time_train_predict_RF[0][5]
df_res.loc["Prediction Accuracy nr ", "RF ({} trees)".format(time_train_predict_RF[0][0])] = time_train_predict_RF[0][6]

df_res.loc["Confusion matrix nr ", "RF ({} trees)".format(time_train_predict_RF[0][0])] = str(cm_nr_RF)




df_res.loc["Training Time (sec)", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = time_train_predict_KN[0][1]
df_res.loc["Training Accuracy", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = time_train_predict_KN[0][2]
df_res.loc["Prediction Time (sec)", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = time_train_predict_KN[0][3]
df_res.loc["Prediction Accuracy", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = time_train_predict_KN[0][4]

df_res.loc["Confusion matrix", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = str(cm_KN)

df_res.loc["Prediction Time nr (sec)", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = time_train_predict_KN[0][5]
df_res.loc["Prediction Accuracy nr ", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = time_train_predict_KN[0][6]

df_res.loc["Confusion matrix nr ", "KN ({} neighbors)".format(time_train_predict_KN[0][0])] = str(cm_nr_KN)


df_res = df_res.fillna('')
df_res.to_excel("untiteld.xlsx")
df_res
'''























































































































