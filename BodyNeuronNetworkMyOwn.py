import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_input_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden_output = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Forward pass
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        return self.output

    def backward(self, X, y, learning_rate):
        # Backpropagation
        error = y - self.output
        d_output = error
        d_hidden_output = np.dot(d_output, self.weights_hidden_output.T)
        d_hidden_input = d_hidden_output * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_output.T, d_output)
        self.bias_hidden_output += learning_rate * np.sum(d_output, axis=0)
        self.weights_input_hidden += learning_rate * np.dot(X.T, d_hidden_input)
        self.bias_input_hidden += learning_rate * np.sum(d_hidden_input, axis=0)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
            loss = np.mean(np.square(y - output))
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')


# Generate sample data
num_samples = 100
num_features = 8

# Generate normal data
normal_data = np.random.normal(loc=0, scale=1, size=(num_samples // 2, num_features))

# Generate abnormal data
abnormal_data = np.random.uniform(low=-2, high=2, size=(num_samples // 2, num_features))

# Create labels (0 for normal, 1 for abnormal)
labels = np.concatenate([np.zeros(num_samples // 2), np.ones(num_samples // 2)])

# Combine normal and abnormal data
data = np.concatenate([normal_data, abnormal_data])

# Shuffle data and labels
p = np.random.permutation(num_samples)
data = data[p]
labels = labels[p]

# Split data into training and testing sets
train_ratio = 0.8
num_train = int(train_ratio * num_samples)
x_train, x_test = data[:num_train], data[num_train:]
y_train, y_test = labels[:num_train], labels[num_train:]

# Initialize and train the neural network
input_size = num_features
hidden_size = 16
output_size = 1
learning_rate = 0.01
epochs = 1000

nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(x_train, y_train, epochs, learning_rate)
