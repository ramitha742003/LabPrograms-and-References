# Build an Artificial Neural Network by implementing the Backpropagation algorithm and test the same using appropriate data sets.

import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) #X: 3 samples with 2 features each.
y = np.array(([92], [86], [89]), dtype=float) #y: Corresponding target output values.

X = X / np.amax(X, axis=0) #X is normalized column-wise (feature-wise) to scale each feature to [0, 1].
y = y / 100 #y is scaled to [0, 1] by dividing by 100.

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) #Sigmoid squashes input into the (0, 1) range.

def derivatives_sigmoid(x):
    return x * (1 - x)          #Used during backpropagation to compute the gradient.

epoch = 5000            #epoch: Number of training iterations.
lr = 0.1                #lr: Learning rate for weight updates.
inputlayer_neurons = 2   #inputlayer_neurons: Number of features (columns in X).
hiddenlayer_neurons = 3 #hiddenlayer_neurons: Number of neurons in the hidden layer.
output_neurons = 1     #output_neurons: Single output neuron (for regression or binary classification).

wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons)) #wh: Weights between input and hidden layer
bh = np.random.uniform(size=(1, hiddenlayer_neurons))                  #bh: Bias for hidden layer
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))   #wout: Weights between hidden and output layer
bout = np.random.uniform(size=(1, output_neurons))                     #bout: Bias for output layer 

for i in range(epoch):                           #Iterates through epoch times to train the network using forward and backward propagation.
    hinp1 = np.dot(X, wh)                        #hinp1: Raw input to the hidden layer (dot product)
    hinp = hinp1 + bh                            #hinp: Hidden layer input after adding bias 
    hlayer_act = sigmoid(hinp)                   #hlayer_act: Activated hidden layer output 
    outinp1 = np.dot(hlayer_act, wout)           #outinp1: Raw input to output layer
    outinp = outinp1 + bout                      # outinp: Output layer input after adding bias
    output = sigmoid(outinp)                     #output: Final output of the network after sigmoid activation

EO = y - output                                #EO: Error between predicted output and actual output.
outgrad = derivatives_sigmoid(output)          #outgrad: Gradient of the sigmoid at output layer.
d_output = EO * outgrad                        #d_output: Element-wise product of error and gradient — delta for output layer.

EH = d_output.dot(wout.T)                     #Propagating the error back to the hidden layer using the output weights (wout).

hiddengrad = derivatives_sigmoid(hlayer_act)        #hiddengrad: Derivative of sigmoid at hidden layer.
d_hiddenlayer = EH * hiddengrad                     #d_hiddenlayer: Delta for hidden layer using error propagated from the output.
wout += hlayer_act.T.dot(d_output) * lr               #Gradient Descent Update using calculated deltas and learning rate lr.
wh += X.T.dot(d_hiddenlayer) * lr

print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n", output)
