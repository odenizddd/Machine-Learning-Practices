import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 12, 34, 56, 58, 62, 65, 97])
y = np.array([1, 2, 3, 4, 6, 23, 45, 46, 47, 48, 90, 98])

w = 0.5
b = 0
lr = 0.00001

for i in range(10000):
    w, b = w - lr * sum([ (w * x[i] + b - y[i]) * x[i] for i in range(len(x)) ]) / len(x), b - lr * sum([ (w * x[i] + b - y[i]) for i in range(len(x)) ]) / len(x)

plt.scatter(x, y)
plt.plot(x, [w * i + b for i in x])
plt.show()
