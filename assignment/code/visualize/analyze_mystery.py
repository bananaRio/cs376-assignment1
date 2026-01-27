import numpy as np
import matplotlib.pyplot as plt

X = np.load("mysterydata/mysterydata2.npy")
for i in range(9):
    # option 1
    X_1 = np.sqrt(X[:,:,i])
    plt.imsave("vis2-op1_%d.png" % i, X_1, cmap='plasma')

for i in range(9):
    # option 2: log correction (log1p is safer to avoid log(0))
    X_2 = np.log1p(X[:,:,i])
    plt.imsave("vis2-op2_%d.png" % i, X_2, cmap='plasma')
