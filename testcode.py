import numpy as np
import random

x = [np.array([1,2]), np.array([3,4,5])]
np.save("test.npy",x)
print(np.load("test.npy"))
