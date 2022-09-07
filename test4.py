import numpy as np

x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.reshape(x,[5,2])
print(y)

y=x.reshape([5,2])
print(y)