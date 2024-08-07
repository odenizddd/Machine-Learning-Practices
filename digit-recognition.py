import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data():
    mnist = load_digits()
    X = mnist.data
    Y = mnist.target
    images = mnist.images
    X /= 16.0
    Y = np.eye(10)[Y]
    return train_test_split(X, Y, images, test_size=0.2, random_state=42)

def initialize_params(layer_sizes):
    params = {}
    for i in range(1, len(layer_sizes)):
        params['W' + str(i)] = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * np.sqrt(2. / layer_sizes[i-1])
        params['B' + str(i)] = np.zeros((layer_sizes[i], 1))
    return params

def softmax(Z):
    Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return Z / np.sum(Z, axis=0, keepdims=True)


def forward_propogation(params, X):
    layers = len(params)//2
    values = {'A0': X.T}
    for i in range(1, layers+1):
        Z = np.dot(params['W'+str(i)], values['A'+str(i-1)]) + params['B'+str(i)]
        values['Z'+str(i)] = Z
        if i == layers:
            values['A'+str(i)] = softmax(Z)
        else:
            values['A'+str(i)] = np.maximum(0, Z)
    return values

def compute_cost(values, Y_true):
    Y_pred = values['A'+str(len(values)//2)]
    return -np.sum(Y_true * np.log(Y_pred.T + 1e-8)) / Y_true.shape[0]

def back_propogation(params, values, Y_true):
    layers = len(params)//2
    dZ = values['A'+str(layers)] - Y_true.T
    m = Y_true.shape[0]
    grads = {}
    grads['W'+str(layers)] = 1 / m * np.dot(dZ, values['A'+str(layers-1)].T)
    grads['B'+str(layers)] = 1 / m * np.sum(dZ, axis=1, keepdims=1)
    for i in range(layers-1, 0, -1):
        dA = np.dot(params['W'+str(i+1)].T, dZ)
        dZ = np.where(values['Z'+str(i)]>0, 1, 0) * dA
        grads['W'+str(i)] = 1 / m * np.dot(dZ, values['A'+str(i-1)].T)
        grads['B'+str(i)] = 1 / m * np.sum(dZ, axis=1, keepdims=1)
    return grads

def update_params(params, grads, learning_rate):
    layers = len(params)//2
    for i in range(1, layers+1):
        params['W'+str(i)] -= learning_rate * grads['W'+str(i)]
        params['B'+str(i)] -= learning_rate * grads['B'+str(i)]

layer_sizes = [64, 16, 16, 10]
learning_rate = 0.03
num_iters = 1000

X_train, X_test, Y_train, Y_test, images_train, images_test = load_data()
params = initialize_params(layer_sizes)

for i in range(num_iters):
    values = forward_propogation(params, X_train)
    cost = compute_cost(values, Y_train)
    grads = back_propogation(params, values, Y_train)
    update_params(params, grads, learning_rate)
    if i % 100 == 0:
        print(f"Cost at iteration {i}: {cost}")

# Evaluation
values = forward_propogation(params, X_test)
labels = np.argmax(Y_test, axis=1)
predictions = np.argmax(values['A'+str(len(layer_sizes)-1)], axis=0)
accuracy = np.mean(labels == predictions) * 100
print(f"Accuract: {accuracy}")

# Display some examples
fig, axes = plt.subplots(2, 10, figsize=(16, 6))
for i in range(20):
    ax = axes[i//10, i%10]
    ax.imshow(images_test[i], cmap="Greens")
    ax.set_axis_off()
    ax.set_title(f"{labels[i]}:{predictions[i]}")
plt.tight_layout()
plt.show()