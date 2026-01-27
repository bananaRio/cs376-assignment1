#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb


def colormapArray(X, colors):
    """
    Basically plt.imsave but return a matrix instead

    Given:
        a HxW matrix X
        a Nx3 color map of colors in [0,1] [R,G,B]
    Outputs:
        a HxW uint8 image using the given colormap. See the Bewares
    """
    print("x.dtype: ", X.dtype)
    print("x.shape: ", X.shape)
    print("================================")

    N = colors.shape[0]
    vmin = np.nanmin(X)
    vmax = np.nanmax(X)
    print("vmin: ", vmin)
    print("vmax: ", vmax)
    finite = np.isfinite(X)
    
    if vmax == vmin:
        idx = np.full(X.shape, N // 2, dtype=np.int32)
    else:
        normalized = (X - vmin) / (vmax - vmin)
        idx = (N - 1) * normalized
        # np.clip clamps(limits) every element of array into range
        # [min, max] in the first parameter
        idx = np.clip(idx, 0, N - 1)
        # whatever x is not finite, set that to N // 2
        idx[~finite] = N // 2
        idx = idx.astype(np.int32) # idx = float so needs to be converted

    rgb_float = colors[idx]

    #convert [0,1] to [0, 255] uint8
    print("rgb float shape: ", rgb_float.shape)
    rgb_uint8 = (rgb_float * 255).astype(np.uint8)
    return rgb_uint8


if __name__ == "__main__":
    colors = np.load("mysterydata/colors.npy")
    data = np.load("mysterydata/mysterydata4.npy")

    # process all 9 channels
    for i in range(9):
        result = colormapArray(data[:,:,i], colors)
        plt.imsave("vis3.3_%d.png" %i, result)
