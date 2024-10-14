import numpy as np

class LogisticRegression():
    #Put in docstring
    
    def __init__(self, random_state):
        # self.weights = 0.0
        self.bias = 0.0
        self.random_state = np.random.seed(random_state)
        
    #Defining our sigmoid function to use in our y_hat calculations
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    #Defining our y_hat function
    def y_hat_calc(self, weights, X, bias):
        z = np.dot(X, weights) + bias
        return self.sigmoid(z)
    
    #Defining our fit/training function:
    def fit(self, X, y, learning_rate = 0.1, n_iter = 250):
        #Add docstrings for this function
        
        #Initializing the weights to have a shape of X's features
        self.weights = np.zeros(shape=np.shape(X[0]))
        m = X.shape[0] #number of samples
        
        #Iterating through epochs to train the model
        for epochs in range(n_iter):
            y_hat = self.y_hat_calc(self.weights, X, self.bias)
            #Getting the loss gradients from our first calculation
            loss_grad_w = np.dot(X.T, (y_hat - y )) / m
            loss_grad_b = np.sum(y_hat - y) / m
            
            self.weights -= (learning_rate * loss_grad_w)
            self.bias -= (learning_rate * loss_grad_b)
    
    #y_prediction function      
    def prediction(self, X_test):
        #Add docstring
        y_hat = self.y_hat_calc(self.weights, X_test, self.bias)
        y_preds = np.where(y_hat >= 0.5, 1, 0)
        return y_preds