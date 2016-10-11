from tools import dropout, add_bias, confirm
import numpy as np
import collections
import math

all = ["backpropagation", "scaled_conjugate_gradient", "scipyoptimize", "resilient_backpropagation"]


def backpropagation(network, trainingset, ERROR_LIMIT = 1e-3, learning_rate = 0.03, momentum_factor = 0.9, max_iterations = ()  ):
    
    assert trainingset[0].features.shape[0] == network.n_inputs, \
            "ERROR: input size varies from the defined input setting"
    
    assert trainingset[0].targets.shape[0]  == network.layers[-1][0], \
            "ERROR: output size varies from the defined output setting"
    
    
    training_data              = np.array( [instance.features for instance in trainingset ] )
    training_targets           = np.array( [instance.targets  for instance in trainingset ] )
                            
    layer_indexes              = range( len(network.layers) )[::-1]    # reversed
    momentum                   = collections.defaultdict( int )
    epoch                      = 0
    
    input_signals, derivatives = network.update( training_data, trace=True )
    
    out                        = input_signals[-1]
    error                      = network.cost_function(out, training_targets )
    cost_derivative            = network.cost_function(out, training_targets, derivative=True).T
    delta                      = cost_derivative * derivatives[-1]
    
    while error > ERROR_LIMIT and epoch < max_iterations:
        epoch += 1
        
        for i in layer_indexes:
            # Loop over the weight layers in reversed order to calculate the deltas
            
            # perform dropout
            dropped = dropout( 
                        input_signals[i], 
                        # dropout probability
                        network.hidden_layer_dropout if i > 0 else network.input_layer_dropout
                    )
            
            # calculate the weight change
            dW = -learning_rate * np.dot( delta, add_bias(dropped) ).T + momentum_factor * momentum[i]
            
            if i != 0:
                """Do not calculate the delta unnecessarily."""
                # Skip the bias weight
                weight_delta = np.dot( network.weights[ i ][1:,:], delta )
    
                # Calculate the delta for the subsequent layer
                delta = weight_delta * derivatives[i-1]
            
            # Store the momentum
            momentum[i] = dW
                                
            # Update the weights
            network.weights[ i ] += dW
        #end weight adjustment loop
        
        input_signals, derivatives = network.update( training_data, trace=True )
        out                        = input_signals[-1]
        error                      = network.cost_function(out, training_targets )
        cost_derivative            = network.cost_function(out, training_targets, derivative=True).T
        delta                      = cost_derivative * derivatives[-1]
        
        
        if epoch%1000==0:
            # Show the current training status
            print "[training] Current error:", error, "\tEpoch:", epoch
    
    print "[training] Finished:"
    print "[training]   Converged to error bound (%.4g) with error %.4g." % ( ERROR_LIMIT, error )
    print "[training]   Trained for %d epochs." % epoch
    
    if network.save_trained_network and confirm( promt = "Do you wish to store the trained network?" ):
        network.save_to_file()
# end backprop

def resilient_backpropagation(network, trainingset, ERROR_LIMIT=1e-3, max_iterations = (), weight_step_max = 50., weight_step_min = 0., start_step = 0.5, learn_max = 1.2, learn_min = 0.5 ):
    # Implemented according to iRprop+ 
    # http://sci2s.ugr.es/keel/pdf/algorithm/articulo/2003-Neuro-Igel-IRprop+.pdf
    assert network.input_layer_dropout == 0 and network.hidden_layer_dropout == 0, \
            "ERROR: dropout should not be used with resilient backpropagation"
    
    assert trainingset[0].features.shape[0] == network.n_inputs, \
            "ERROR: input size varies from the defined input setting"
    
    assert trainingset[0].targets.shape[0]  == network.layers[-1][0], \
            "ERROR: output size varies from the defined output setting"
    
    training_data              = np.array( [instance.features for instance in trainingset ] )
    training_targets           = np.array( [instance.targets  for instance in trainingset ] )
    
    # Data structure to store the previous derivative
    previous_dEdW                  = [ 1 ] * len( network.weights )
    
    # Storing the current / previous weight step size
    weight_step                = [ np.full( weight_layer.shape, start_step ) for weight_layer in network.weights ]
    
    # Storing the current / previous weight update
    dW                         = [  np.ones(shape=weight_layer.shape) for weight_layer in network.weights ]
    
    
    input_signals, derivatives = network.update( training_data, trace=True )
    out                        = input_signals[-1]
    cost_derivative            = network.cost_function(out, training_targets, derivative=True).T
    delta                      = cost_derivative * derivatives[-1]
    error                      = network.cost_function(out, training_targets )
    
    layer_indexes              = range( len(network.layers) )[::-1] # reversed
    prev_error                   = ( )                             # inf
    epoch                      = 0
    
    while error > ERROR_LIMIT and epoch < max_iterations:
        epoch       += 1
        
        for i in layer_indexes:
            # Loop over the weight layers in reversed order to calculate the deltas
                   
            # Calculate the delta with respect to the weights
            dEdW = np.dot( delta, add_bias(input_signals[i]) ).T
            
            if i != 0:
                """Do not calculate the delta unnecessarily."""
                # Skip the bias weight
                weight_delta = np.dot( network.weights[ i ][1:,:], delta )
    
                # Calculate the delta for the subsequent layer
                delta = weight_delta * derivatives[i-1]
            
            
            # Calculate sign changes and note where they have changed
            diffs            = np.multiply( dEdW, previous_dEdW[i] )
            pos_indexes      = np.where( diffs > 0 )
            neg_indexes      = np.where( diffs < 0 )
            zero_indexes     = np.where( diffs == 0 )
            
            
            # positive
            if np.any(pos_indexes):
                # Calculate the weight step size
                weight_step[i][pos_indexes] = np.minimum( weight_step[i][pos_indexes] * learn_max, weight_step_max )
                
                # Calculate the weight step direction
                dW[i][pos_indexes] = np.multiply( -np.sign( dEdW[pos_indexes] ), weight_step[i][pos_indexes] )
                
                # Apply the weight deltas
                network.weights[i][ pos_indexes ] += dW[i][pos_indexes]
            
            # negative
            if np.any(neg_indexes):
                weight_step[i][neg_indexes] = np.maximum( weight_step[i][neg_indexes] * learn_min, weight_step_min )
                
                if error > prev_error:
                    # iRprop+ version of resilient backpropagation
                    network.weights[i][ neg_indexes ] -= dW[i][neg_indexes] # backtrack
                
                dEdW[ neg_indexes ] = 0
            
            # zeros
            if np.any(zero_indexes):
                dW[i][zero_indexes] = np.multiply( -np.sign( dEdW[zero_indexes] ), weight_step[i][zero_indexes] )
                network.weights[i][ zero_indexes ] += dW[i][zero_indexes]
            
            # Store the previous weight step
            previous_dEdW[i] = dEdW
        #end weight adjustment loop
        
        prev_error                 = error
        
        input_signals, derivatives = network.update( training_data, trace=True )
        out                        = input_signals[-1]
        cost_derivative            = network.cost_function(out, training_targets, derivative=True).T
        delta                      = cost_derivative * derivatives[-1]
        error                      = network.cost_function(out, training_targets )
        
        if epoch%1000==0:
            # Show the current training status
            print "[training] Current error:", error, "\tEpoch:", epoch

    print "[training] Finished:"
    print "[training]   Converged to error bound (%.4g) with error %.4g." % ( ERROR_LIMIT, error )
    print "[training]   Trained for %d epochs." % epoch
    
    if network.save_trained_network and confirm( promt = "Do you wish to store the trained network?" ):
        network.save_to_file()
# end backprop

def scipyoptimize(network, trainingset, method = "Newton-CG", ERROR_LIMIT = 1e-6, max_iterations = ()  ):
    from scipy.optimize import minimize
    
    training_data        = np.array( [instance.features for instance in trainingset ] )
    training_targets     = np.array( [instance.targets  for instance in trainingset ] )
    minimization_options = {}
    
    if max_iterations < ():
        minimization_options["maxiter"] = max_iterations
        
    results = minimize( 
        network.error,                                     # The function we are minimizing
        network.get_weights(),                             # The vector (parameters) we are minimizing
        args    = (training_data, training_targets),    # Additional arguments to the error and gradient function
        method  = method,                               # The minimization strategy specified by the user
        jac     = network.gradient,                        # The gradient calculating function
        tol     = ERROR_LIMIT,                          # The error limit
        options = minimization_options,                 # Additional options
    )
    
    network.weights = network.unpack( results.x )
    
    
    if not results.success:
        print "[training] WARNING:", results.message
        print "[training]   Converged to error bound (%.4g) with error %.4g." % ( ERROR_LIMIT, results.fun )
    else:
        print "[training] Finished:"
        print "[training]   Converged to error bound (%.4g) with error %.4g." % ( ERROR_LIMIT, results.fun )
        
        if network.save_trained_network and confirm( promt = "Do you wish to store the trained network?" ):
            network.save_to_file()
#end

def scaled_conjugate_gradient(network, trainingset, ERROR_LIMIT = 1e-6, max_iterations = () ):
    # Implemented according to the paper by Martin F. Moller
    # http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.3391
    
    assert network.input_layer_dropout == 0 and network.hidden_layer_dropout == 0, \
            "ERROR: dropout should not be used with scaled conjugated gradients training"
            
    assert trainingset[0].features.shape[0] == network.n_inputs, \
            "ERROR: input size varies from the defined input setting"
    
    assert trainingset[0].targets.shape[0]  == network.layers[-1][0], \
            "ERROR: output size varies from the defined output setting"
    
    
    training_data       = np.array( [instance.features for instance in trainingset ] )
    training_targets    = np.array( [instance.targets  for instance in trainingset ] )
    

    ## Variables
    sigma0              = 1.e-6
    lamb                = 1.e-6
    lamb_               = 0

    vector              = network.get_weights() # The (weight) vector we will use SCG to optimalize
    N                   = len(vector)
    grad_new            = -network.gradient( vector, training_data, training_targets )
    r_new               = grad_new
    # end

    success             = True
    k                   = 0
    while k < max_iterations:
        k               += 1
        r               = np.copy( r_new     )
        grad            = np.copy( grad_new  )
        mu              = np.dot(  grad,grad )
    
        if success:
            success     = False
            sigma       = sigma0 / math.sqrt(mu)
            s           = (network.gradient(vector+sigma*grad, training_data, training_targets)-network.gradient(vector,training_data, training_targets))/sigma
            delta       = np.dot( grad.T, s )
        #end
    
        # scale s
        zetta           = lamb-lamb_
        s               += zetta*grad
        delta           += zetta*mu
    
        if delta < 0:
            s           += (lamb - 2*delta/mu)*grad
            lamb_       = 2*(lamb - delta/mu)
            delta       -= lamb*mu
            delta       *= -1
            lamb        = lamb_
        #end
    
        phi             = np.dot( grad.T,r )
        alpha           = phi/delta
    
        vector_new      = vector+alpha*grad
        f_old, f_new    = network.error(vector,training_data, training_targets), network.error(vector_new,training_data, training_targets)
    
        comparison      = 2 * delta * (f_old - f_new)/np.power( phi, 2 )
        
        if comparison >= 0:
            if f_new < ERROR_LIMIT: 
                break # done!
        
            vector      = vector_new
            f_old       = f_new
            r_new       = -network.gradient( vector, training_data, training_targets )
        
            success     = True
            lamb_       = 0
        
            if k % N == 0:
                grad_new = r_new
            else:
                beta    = (np.dot( r_new, r_new ) - np.dot( r_new, r ))/phi
                grad_new = r_new + beta * grad
        
            if comparison > 0.75:
                lamb    = 0.5 * lamb
        else:
            lamb_       = lamb
        # end 
    
        if comparison < 0.25: 
            lamb        = 4 * lamb
    
        if k%1000==0:
            print "[training] Current error:", f_new, "\tEpoch:", k
    #end
    
    network.weights = network.unpack( np.array(vector_new) )
    
    print "[training] Finished:"
    print "[training]   Converged to error bound (%.4g) with error %.4g." % ( ERROR_LIMIT, f_new )
    print "[training]   Trained for %d epochs." % k
    
    
    if network.save_trained_network and confirm( promt = "Do you wish to store the trained network?" ):
        network.save_to_file()
#end scg