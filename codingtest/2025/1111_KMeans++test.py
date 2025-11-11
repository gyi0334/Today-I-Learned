import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

print(fruits.shape)
print(fruits_2d.shape)