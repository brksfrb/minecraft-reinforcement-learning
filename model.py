import math, random, json


def tanh(x):
    return math.tanh(x)  # built-in tanh, outputs -1 to 1

def tanh_derivative(y):
    return 1 - y**2  # note: derivative uses the output of tanh, not input

class NeuralNetwork:
    def __init__(self):
        self.input_size = 3
        self.hidden_size = 8
        self.output_size = 3

        # Random weights
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(self.hidden_size)]
                                     for _ in range(self.input_size)]
        self.weights_hidden_output = [[random.uniform(-1, 1) for _ in range(self.output_size)]
                                      for _ in range(self.hidden_size)]

        # Biases
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(self.hidden_size)]
        self.bias_output = [random.uniform(-1, 1) for _ in range(self.output_size)]

        self.learning_rate = 0.1

    def forward(self, inputs):
        self.inputs = inputs

        self.hidden_outputs = []
        for j in range(self.hidden_size):
            total = self.bias_hidden[j]
            for i in range(self.input_size):
                total += inputs[i] * self.weights_input_hidden[i][j]
            self.hidden_outputs.append(tanh(total))

        # Output layer
        self.final_outputs = []
        for k in range(self.output_size):
            total = self.bias_output[k]
            for j in range(self.hidden_size):
                total += self.hidden_outputs[j] * self.weights_hidden_output[j][k]
            self.final_outputs.append(tanh(total))

        return self.final_outputs

    def decide_action(self, state, epsilon=0.1):
        if random.random() < epsilon:  # explore
            return random.choice(["move_forward", "rotate_15_right", "rotate_15_left"])

        # exploit (normal network prediction)
        outputs = self.forward(state)
        action_index = outputs.index(max(outputs))
        actions = ["move_forward", "rotate_15_right", "rotate_15_left"]
        return actions[action_index]

    # Simple RL weight update: policy gradient-like using reward
    def train(self, state, reward):
        outputs = self.forward(state)  # compute forward pass
        # now self.inputs, self.hidden_outputs, self.final_outputs are set
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] += self.learning_rate * reward * self.inputs[i] * tanh_derivative(
                    self.hidden_outputs[j])
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.weights_hidden_output[j][k] += self.learning_rate * reward * self.hidden_outputs[
                    j] * tanh_derivative(self.final_outputs[k])
        for j in range(self.hidden_size):
            self.bias_hidden[j] += self.learning_rate * reward * tanh_derivative(self.hidden_outputs[j])
        for k in range(self.output_size):
            self.bias_output[k] += self.learning_rate * reward * tanh_derivative(self.final_outputs[k])

    def save(self, filename):
        model_data = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "weights_input_hidden": self.weights_input_hidden,
            "weights_hidden_output": self.weights_hidden_output,
            "bias_hidden": self.bias_hidden,
            "bias_output": self.bias_output
        }
        with open(filename, "w") as f:
            json.dump(model_data, f)
        print(f"Model saved to {filename}")

    def load(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        self.input_size = data["input_size"]
        self.hidden_size = data["hidden_size"]
        self.output_size = data["output_size"]
        self.learning_rate = data["learning_rate"]
        self.weights_input_hidden = data["weights_input_hidden"]
        self.weights_hidden_output = data["weights_hidden_output"]
        self.bias_hidden = data["bias_hidden"]
        self.bias_output = data["bias_output"]
        print(f"Model loaded from {filename}")