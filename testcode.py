import numpy as np
import random
x = np.arange(5)
y=np.arange(0,10,2)
y=y.reshape((5,1))
x=x.reshape((5,1))
print(np.dot(x,y))
