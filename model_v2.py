import numpy as np
import math
import random

def relu(x):
    return max(0, x)
def tanh(x):
    return np.tanh(x)
def relu_derivative(x):
    return 1 if x > 0 else 0
def tanh_derivative(x):
    return 1 - math.tanh(x)**2

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = []
        self.weights_hidden_output = []
        self.bias_hidden = []
        self.bias_output = []
        self.dataset_in = [[0,0], [0,1], [1,0], [1,1]]
        self.dataset_out = [[0], [1], [1], [0]]
        self.lr = 0.01
        self.epochs = 1000
        for i in range(hidden_size):
            self.weights_input_hidden.append([])
            self.bias_hidden.append([])
            self.bias_hidden[i].append(random.randint(-100, 100) / 100)
            for j in range(input_size):
                self.weights_input_hidden[i].append(.1)
                self.weights_input_hidden[i][j] = random.randint(-100, 100) / 100


        for i in range(output_size):
            self.weights_hidden_output.append([])
            self.bias_output.append([])
            self.bias_output[i] = random.randint(-100, 100) / 100
            for j in range(input_size):
                self.weights_hidden_output[i].append([])
                self.weights_hidden_output[i][j] = random.randint(-100, 100) / 100

    def forward(self, inputs):
        self.hidden = []
        self.output = []
        self.output_copy = []
        self.hidden_copy = []
        for i in range(self.hidden_size):
            self.hidden.append(.1)
            self.hidden[i] = self.bias_hidden[i]
            for j in range(self.input_size):
                print(self.weights_input_hidden[i][j])
                print(inputs[j])
                print(self.hidden[i])
                self.hidden[i][0] += self.weights_input_hidden[i][j] * inputs[j]

            self.hidden_copy[i] = self.hidden[i]
            self.hidden[i] = relu(self.hidden[i])

        for i in range(self.output_size):
            self.output[i] = self.bias_output[i]
            for j in range(self.hidden_size):
                self.output[i] += self.weights_hidden_output[i][j] * self.hidden[j]

            self.output_copy[i] = self.output[i]
            self.output[i] = tanh(self.hidden[i])


    def backwards_pass(self):
        self.output_error = []
        self.hidden_error = []
        for i in range(self.output_size):
            self.output_error[i] = self.output[i] - self.dataset_out[i]
            self.bias_output[i] -= self.output_error[i] * self.lr
            for j in range(self.hidden_size):
                self.weights_hidden_output[i][j] -= self.output_error[i] * self.lr * relu_derivative(self.output_copy[i])

        for i in range(self.hidden_size):
            self.hidden_error[i] = self.output_error[j] * self.weights_hidden_output[i][j]
            self.bias_hidden[i] = self.hidden_error[i] * self.lr
            for j in range(self.output_size):
                self.weights_input_hidden[i][j] -= self.hidden_error[i] * self.lr * relu_derivative(self.hidden_copy[i])

nn = NeuralNetwork(input_size=2, output_size=1, hidden_size=4)
for epoch in range(nn.epochs):
    for subepoch in range(len(nn.dataset_in)):
        nn.forward(nn.dataset_in[subepoch])
        nn.backwards_pass()
        print("Epoch:", epoch + 1)