import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


#Doing this all as one function first, then building this into a class
def log_regressor(X_train, y_train, X_test, y_test):
    #initializing the weights and bias term
    weights = np.zeros(np.shape(X_train[1]))
    bias_term = 0.0
    #Training our model
    weights, bias_term = stochastic_gradient_descent(X_train, y_train, weights=weights, bias_term=bias_term, learning_rate=0.1)
    

    #Making a prediction and testing
    y_hat = sigmoid_function(np.dot(X_test, weights) + bias_term)
    y_preds = np.where(y_hat >= 0.5, 1, 0)
    # print(y_preds)
    print(f'\nBENS ACCURACY SCORE:\t {accuracy_score(y_test, y_preds)}')
    print(f'BENS F1 SCORE:\t {f1_score(y_test, y_preds)}')


#Defining our y_hat computation
def y_hat_function(weights, input_vector, bias_term): #X is the input vector
    #Getting our z to pass into our sigmoid function
    z = np.dot(a=input_vector, b=weights) + bias_term
    y_hat = sigmoid_function(z)
    return y_hat

#Defining our sigmoid function
def sigmoid_function(z):
    return 1/( 1 +np.exp(-z))

#NOTE: We might not need this because we have the derived gradient
#defining our loss function
def cross_entropy(y_hat, y):
    loss = -(y*np.log(y_hat) + (1 - y)*np.log(y_hat))
    return loss

#defining our stochastic gradient descent function
def stochastic_gradient_descent(X, y, weights, bias_term, learning_rate = 0.1, n_iter = 250):  
    for epoch in range(n_iter):
        #m = number of samples
        m = X.shape[0]
        #computing y_hat for this instance
        y_hat = sigmoid_function(np.dot(X, weights) + bias_term)
        #Calculating our loss for our print statement
        # loss = cross_entropy(y_hat, y)
        #Computing our loss gradients
        loss_gradient_w = np.dot(X.T, (y_hat - y)) / m  
        loss_gradient_b = np.sum(y_hat - y) / m


        weights -=  (learning_rate*loss_gradient_w)
        bias_term -=  learning_rate*loss_gradient_b
            
        ## Information about the training if desired
        # print(f'Result of epoch #{epoch + 1}')
        # print(f'Loss: {(loss_gradient_w)}')
        # print(f'THETA: {weights}')
        # print(f'Bias Term: {weights}\n')

    return weights, bias_term



if __name__ == "__main__":
    # Creating our sample data for prototyping
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)  
    log_regressor(X_train, y_train, X_test, y_test)