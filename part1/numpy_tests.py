import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

# MAX OVER AN AXIS
# x = np.array([i for i in range(9)]).reshape(3,3)
# print(x)
# y = np.zeros_like(x)
# c = np.array([np.max(x, axis=1)]).reshape(-1, 1) # better alternative is to use keepdim=True
#
# print(x-c)
#
# # SUM OVER AN AXIS
# print(np.sum(x, axis=1))
# z = np.exp(x-c)
# d = np.array([np.sum(z, axis=1)]).reshape(-1, 1) # better alternative is to use keepdim=True

# print(z/d)

# row = np.array([0, 2, 1, 0])
# col = np.array([0, 2, 1, 2])
# data = np.array([4, 5, 7, 9])

# SPARSE COO-MATRIX
# print(sparse.coo_matrix(([1]*4, (row, col)), shape=(3, 4)).toarray())
# print(np.array([1]*5))

# EFFECT OF TEMPERATURE PARAMETER

x_i = np.array([1]*30).reshape(2, 15)
print(x_i)
theta = np.array([[1]*15,[2]*15,[3]*15,[4]*15] )
# theta = np.diag(np.linspace(0, 1, 15))
print(theta)
temp = [0.5, 5, 10]
z = x_i@theta.T

for tau in temp:
    logit = z/tau
    logit = logit - np.amax(logit, axis=1, keepdims=True)

    softmax = np.exp(logit)
    softmax = softmax / np.sum(softmax, axis=1, keepdims=True)
    print(np.var(softmax))

    plt.plot(softmax[0],marker='o', label=tau)
    print(f'{tau}: probabilities: {softmax}')

plt.legend()
plt.show()