import numpy as np
from utils import train_test_split
import sys
print(sys.version)

def sigmoid(z):
    """
    computes the sigmoid of z.
    """
    return 1 / (1 + np.e**(-z))

def loss(X,y,theta):
    """
    Loss function for logistic regression
    """
    
    z = np.dot(X,theta)
    h = sigmoid(z)
    
    return (1/m)*sum(-y * np.log(h) - (1 - y) * np.log(1 - h))

def compute_cost(X, y, theta, lam = 0):
    
    m = len(y)
    
    J = (1/m) * sum( (-y) * np.log( sigmoid(X.dot(theta)) ) - (1-y) * np.log( 1-sigmoid(X.dot(theta)) ) ) + ( (lam/(2*m)) * sum(theta[1:len(theta)]**2)) 
    
    return J

def gradient_descent(X, y, theta, alpha, epochs, lam = 0):
    m = len(y)
    cost_history = np.zeros([epochs,1])
    
    for i in range(epochs):
        theta = theta*(1-(alpha*(lam/m))) - alpha * (1/m) * np.dot(X.T,(sigmoid(X.dot(theta)) - y))
        
        
        cost_history[i] = compute_cost(X, y, theta, lam)

    return theta, cost_history

def logistic_regression(X,y,theta = None,alpha = .01,epochs=500,lam=0):

    m = len(y) # number of samples

    # declare intercept 
    X = np.append(np.ones([m,1]),X,axis=1)

    # initialize theta parameters if not given
    if theta == None:
        theta = np.zeros([X.shape[1],1])
    
    print(f"Running Gradient Descent with {epochs} iterations and a learning rate of {alpha}...")

    theta, cost_history = gradient_descent(X, y, theta, alpha, epochs, lam)

    print(f"Computed theta parameters are: \n{theta}")
    
    return theta,cost_history


def predict(inputs,theta):
    # adds the intercept value to the data the user is predicting if it is not there
    if inputs.shape[1] != theta.shape[0]:
        inputs = np.append(np.ones([len(inputs),1]),inputs,axis=1)
    return [round(sigmoid(d)[0]) for d in inputs.dot(theta)]

def accuracy(preds,y):
    return 1-(sum(1 for i, j in zip(y, preds) if i != j)/len(y))




data = np.genfromtxt('../../data/binary_class_bank_note.csv',delimiter=',',skip_header=1)
train_X,train_y,test_X,test_y = train_test_split(data)

theta,cost_history = logistic_regression(train_X,train_y,epochs = 1000)

preds = predict(test_X,theta)
acc = accuracy(preds,test_y)
print(f"Accuracy of test set: {acc}")


    