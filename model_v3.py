import numpy as np

# -----------------------
# Activation functions
# -----------------------

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    """Use sigmoid for final output to get 0-1 prediction"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# -----------------------
# Neural Network class
# -----------------------

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        # Initialize weights and biases
        self.weights_input_hidden = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.bias_hidden = np.random.uniform(-1, 1, (hidden_size, 1))

        self.weights_hidden_output = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (output_size, 1))

        # XOR dataset
        self.dataset_in = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19]]
        self.dataset_out = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]

    # -----------------------
    # Forward pass
    # -----------------------
    def forward(self, inputs):
        self.inputs = np.array(inputs).reshape(-1,1)

        # Hidden layer
        self.hidden_raw = np.dot(self.weights_input_hidden, self.inputs) + self.bias_hidden
        self.hidden_activated = relu(self.hidden_raw)

        # Output layer
        self.output_raw = np.dot(self.weights_hidden_output, self.hidden_activated) + self.bias_output
        self.output_activated = sigmoid(self.output_raw)  # sigmoid for 0-1 output

        return self.output_activated

    # -----------------------
    # Backward pass
    # -----------------------
    def backward(self, target):
        target = np.array(target).reshape(-1,1)

        # Output layer
        output_error = self.output_activated - target
        output_delta = output_error * sigmoid_derivative(self.output_raw)

        # Hidden layer
        hidden_error = np.dot(self.weights_hidden_output.T, output_delta)
        hidden_delta = hidden_error * relu_derivative(self.hidden_raw)

        # Update weights and biases
        self.weights_hidden_output -= self.lr * np.dot(output_delta, self.hidden_activated.T)
        self.bias_output -= self.lr * output_delta

        self.weights_input_hidden -= self.lr * np.dot(hidden_delta, self.inputs.T)
        self.bias_hidden -= self.lr * hidden_delta

    # -----------------------
    # Train function
    # -----------------------
    def train(self, epochs=10000):
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(self.dataset_in, self.dataset_out):
                output = self.forward(x)
                self.backward(y)
                total_loss += np.mean((output - np.array(y).reshape(-1,1))**2)
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(self.dataset_in):.4f}")

    # -----------------------
    # Prediction
    # -----------------------
    def predict(self, x):
        output = self.forward(x)
        return 1 if output >= 0.5 else 0  # threshold to 0 or 1

    def predict_raw(self, x):
        output = self.forward(x)
        # output is a numpy array (1x1), get the float value and round to 2 decimals
        return round(float(output), 2)


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    nn = NeuralNetwork(input_size=1, hidden_size=32, output_size=1, lr=0.01)
    nn.train(epochs=5000)

    print("\nXOR Predictions after training:")
    for x in nn.dataset_in:
        print(f"Input: {x}, Predicted Output: {nn.predict(x)}")
