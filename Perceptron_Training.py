#!/usr/bin/env python
# coding: utf-8

import numpy as np

# Input data and expected outputs
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Initial parameters
w1, w2, bias = 0.1, 0.2, 0.1
learning_rate = 0.4
epochs = 6000

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Training the perceptron
for epoch in range(epochs):
    for i in range(len(x)):
        # Compute weighted sum and prediction
        weighted_sum = x[i][0] * w1 + x[i][1] * w2 + bias
        prediction = sigmoid(weighted_sum)
        
        # Calculate error and update weights/bias
        error = y[i] - prediction
        w1 += learning_rate * error * x[i][0]
        w2 += learning_rate * error * x[i][1]
        bias += learning_rate * error

# Display the results
print("Output after training:")
for i in range(len(x)):
    weighted_sum = x[i][0] * w1 + x[i][1] * w2 + bias
    prediction = sigmoid(weighted_sum)
    print(f"Input: {x[i]}, Predicted Output: {prediction:.4f}, Expected Output: {y[i]}")
