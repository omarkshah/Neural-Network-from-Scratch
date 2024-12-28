import numpy as np
import struct
from array import array

# Load the MNIST dataset: https://www.kaggle.com/code/hojjatk/read-mnist-dataset
def read_images_labels(images_filepath, labels_filepath):        
    labels = []
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())        
    
    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())        
    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i][:] = img            
    
    return images, labels

x_train, y_train = read_images_labels("./data/train-images-idx3-ubyte/train-images-idx3-ubyte", "./data/train-labels-idx1-ubyte/train-labels-idx1-ubyte")
x_test, y_test = read_images_labels("./data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", "./data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = x_train.reshape(x_train.shape[0], 784).T
x_test = x_test.reshape(x_test.shape[0], 784).T

#Rectified Linear Unit (ReLu) activation function
def ReLu(Z):
    return np.maximum(Z, 0)

def softmax(Z): 

  
    Z_stable = Z - np.max(Z, axis=0, keepdims=True)  # Numerical stability trick
    expZ = np.exp(Z_stable)
    probabilities = expZ / np.sum(expZ, axis=0, keepdims=True)
    return probabilities


    
    # expSum = np.sum(np.exp(Z), axis=0)
    # probabilities = np.exp(Z) / expSum

    # return probabilities

#forward prop
def forwardProp(W1, W2, b1, b2, X): #weights and biases for both first layer and output layer and the input layer (X) 

    #FIRST layer
    Z1 = W1.dot(X) + b1 
    A1 = ReLu(Z1)

    #SECOND layer (output)
    Z2= W2.dot(A1) + b2
    A2 = softmax(Z2)
    
    return A1, Z1, A2, Z2 #returning results to later update the biases

def backwards(A1, A2, Z1, W2, X, Y):

    Y_onehot = np.zeros((Y.size, Y.max() + 1))
    Y_onehot[np.arange(Y.size), Y] = 1
    Y_onehot = Y_onehot.T

    #second (output) layer
    a_m = np.shape(A1)[0]
    dW2 =  (1/a_m)*(A2 - Y_onehot).dot(A1.T)   
    db2 = (1/a_m) * np.sum(A2 - Y_onehot)

    #first layer
    x_m = np.shape(X)[0]
    dReLu = (Z1 > 0).astype(float)

    dW1 = (1/x_m) * (((W2.T).dot(A2 - Y_onehot) * dReLu).dot(X.T))
    # dW1 = (1/x_m)*( ((W2.T).dot(A2 - Y_onehot)).dot(X.T) * (dReLu)) 
    db1 = (1/x_m) * (np.sum((W2.T).dot(A2 - Y_onehot) * dReLu))

    return dW1, dW2, db1, db2

def  optimize(W1, W2, b1, b2, dW1, dW2, db1, db2, lr):
    W1 = W1 - lr*dW1
    W2 = W2 - lr*dW2
    b1 = b1 - lr*db1
    b2 = b2 - lr*db2

    return W1, W2, b1, b2

def train():

    W1 = np.random.rand(10, 784) - 0.5#hiden layers has 10 neurons (10 rows) each of which need a weight corresponding to each input neuron
    b1 = np.random.rand(10, 1) - 0.5 #random bias specific to each neuron in the hidden layer 

    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
# 
    learning_rate = 1e-3
    epochs = 5
    for i in range(epochs):
        A1, Z1, A2, Z2 = forwardProp(W1, W2, b1, b2, x_train)

        #calculate gradients (backwards prop)
        dW1, dW2, db1, db2 = backwards(A1, A2, Z1, W2, x_train, y_train)

        #update params
        W1, W2, b1, b2 = optimize(W1, W2, b1, b2, dW1, dW2, db1, db2, learning_rate)

        print("Iteration: ", i)
        predictions = np.argmax(A2, 0)
        print(predictions)
        print(np.sum(predictions == y_train) / y_train.size)
        print("-----")
        

train()

#to save just save the weights and biases and use forward prop and A2 is the answer