"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    err = None
    n=len(y)
    np1=np.matmul(X,w)
    err=(np.sum((np1-y)**2))/n

    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """

    #####################################################
    #	TODO 2: Fill in your code here #
    #####################################################		
    w = None
    np1=np.matmul(np.transpose(X),X)
    np2=np.matmul(np.transpose(X),y)
    #print("np1",np1)
    #print("np2",np2)
    np3=np.linalg.inv(np1)
    #print("np3",np3)
    w=np.matmul(np3,np2)
    return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    w = None
    d=len(X[0])
    np1=np.matmul(np.transpose(X),X)
    a,b=np.linalg.eig(np1)
    
    while(np.nanmin(np.absolute(a))<0.00001):
        #print("hey")
        ide=np.identity(d,dtype=float)
        ide=0.1*ide  
        np1=np.add(np1,ide)
        a,b=np.linalg.eig(np1)
        
    np2=np.matmul(np.transpose(X),y)
    #print("np1",np1)
    #print("np2",np2)
    np3=np.linalg.inv(np1)
    #print("np3",np3)
    w=np.matmul(np3,np2)    
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
    w = None
    d=len(X[0])
    np1=np.matmul(np.transpose(X),X)
    a,b=np.linalg.eig(np1)
    ide=np.identity(d,dtype=float)
    ide=lambd*ide  
    np1=np.add(np1,ide)
  
    np2=np.matmul(np.transpose(X),y)
    #print("np1",np1)
    #print("np2",np2)
    np3=np.linalg.inv(np1)
    #print("np3",np3)
    w=np.matmul(np3,np2) 
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    bestlambda = None
    lambdas=[]
    trial_lambda=[]
    l_set=range(-19,20,1)
    for l in l_set:
        trial_lambda.append(10**l)
    for t in trial_lambda:
        w=regularized_linear_regression(Xtrain, ytrain, t)
        error=mean_square_error(w, Xval, yval)
        lambdas.append(error)
    #print("lambdas",lambdas)
    min_ind=np.argmin(lambdas)
    #print("min index",min_ind)
    #print("trial",trial_lambda)
    #print("bestlambda",bestlambda)
    bestlambda=trial_lambda[min_ind]
    #print("bestlambda",bestlambda)
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################		
    np1=X
    #print('1',np1)
    count=2
    while(count<=power):
        
        np2=X**count
        #print('2',np2)
        np1=np.concatenate((np1,np2),axis=1)
        count+=1
    X=np1
    return X


