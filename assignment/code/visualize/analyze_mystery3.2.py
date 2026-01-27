import numpy as np
import matplotlib.pyplot as plt

X = np.load("mysterydata/mysterydata3.npy")
# if isfinite() == False, then the values are NaN (not an number)
# if isfinite() == True,  then the values are +- infinity
# if the mean of isfinite() = 1, good, anythin else = not good
print("fraction of finite values: ", np.mean(np.isfinite(X)), "\n")
print("numbero of invalid values: ", np.sum(~np.isfinite(X)))
for i in range(9):
    
    vmin = np.nanmin(X[:,:,i])
    vmax = np.nanmax(X[:,:,i])
    print(f"channel{i}: vmin = {vmin:.2f} vmax = {vmax:.2f}")
    plt.imsave('vis_3.2_%d.png' %i, X[:,:,i], vmin=vmin, vmax=vmax,cmap='plasma')
    

