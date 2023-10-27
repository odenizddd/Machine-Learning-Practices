import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'x': [0.1, 0.3, 0.6, 0.7],
    'y': [0.2, 0.25, 0.4, 0.7],
})

class NN:
    def __init__(self, dataset):
        self.dataset = dataset
        self.weight = 0.5
        self.bias = 0
        self.predictions = [self.weight * x + self.bias for x in self.dataset['x']]
        
    def update_model(self):
        weight_differentials = [(-(y - pred)) * x for x, y, pred in zip(self.dataset['x'], self.dataset['y'], self.predictions)]
        weight_gradient = sum(weight_differentials) / len(weight_differentials)
        self.weight = self.weight - weight_gradient

        bias_differentials = [(-(y - pred)) for y, pred in zip(self.dataset['y'], self.predictions)]
        bias_gradient = sum(bias_differentials) / len(bias_differentials)
        self.bias = self.bias - bias_gradient

        self.predictions = [self.weight * x + self.bias for x in self.dataset['x']]
        
    def get_error(self):
        errors = [(y - pred) ** 2 for y, pred in zip(self.dataset['y'], self.predictions)]
        return 0.5 * sum(errors)

    def print_nn(self):
        print(f'Weight: {self.weight}')
        print(f'Bias: {self.bias}')
        print("Predictions: ", end='')
        print(self.predictions)

nn = NN(df)

fig, ax = plt.subplots(figsize=(10, 6))

#plot the original data
ax.scatter(nn.dataset['x'], nn.dataset['y'])

ax.axline((0, nn.bias), slope=nn.weight)
ax.scatter(nn.dataset['x'], nn.predictions)
print(nn.get_error())

for i in range(10):
    nn.update_model()

    #plot the current model line and predictions along it
    #ax.axline((0, nn.bias), slope=nn.weight)
    #ax.scatter(nn.dataset['x'], nn.predictions)
    print(nn.get_error())


#plot config
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid()

plt.show()