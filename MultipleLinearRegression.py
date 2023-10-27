import numpy as np

x = np.array([ [1, 1], [1, 2], [2, 1], [2, 2] ])
model = lambda x1, x2: 3 * x1 + 0.5 * x2 + 3

y = np.array([ model(x1, x2) for (x1, x2) in x ])

m, n = x.shape

w = np.zeros((n))
b = 0

lr = 0.2

def cost(w, b, x, y):
    m = x.shape[0]
    total_error = sum([np.dot(w, x[i]) + b - y[i] for i in range(m)])
    return total_error / m


def gradient_j(w, b, x, y, j):
    m = x.shape[0]
    total_sum = sum([ (np.dot(w, x[i]) + b - y[i]) * x[i, j] for i in range(m)])
    return total_sum / m

def gradients_w(w, b, x, y):
    gradients_w = np.zeros(w.shape[0])
    for i in range(m):
        loss = np.dot(w, x[i]) + b - y[i]
        for j in range(n):
            gradients_w[j] += loss * x[i, j]
    return gradients_w / x.shape[0]

def gradient_b(w, b, x, y):
    m = x.shape[0]
    total_sum = sum([ (np.dot(w, x[i]) + b - y[i]) for i in range(m)])
    return total_sum / m

for eras in range(1000):
    #gradients_j = np.array([gradient_j(w, b, x, y, j) for j in range(n)])
    gradients_j = gradients_w(w, b, x, y)
    gradients_b = gradient_b(w, b, x, y)

    w = w - lr * gradients_j
    b = b - lr * gradients_b

print(w, b)
