import numpy as np
import matplotlib.pyplot as plt
from const import *

def smooth(y, width):
  box = np.ones(width)/width
  return np.convolve(y, box, mode='same')

def read_plot():

    image = 1.0 - np.mean(plt.imread("spectrum.png"), axis=2)[:80, :]
    image[-1,:] = 1.0

    weight = np.expand_dims(np.linspace(1.0, 0.0, 80), 1)

    ys = np.sum(image * weight, axis=0) / np.sum(image, axis=0)
    
    ys = ys.repeat(ELONGATION_FACTOR)
   
    ys = smooth(ys, ELONGATION_FACTOR)

    xs = np.linspace(0, MAX_FREQUENCY, len(ys))
    return (xs, ys)


if __name__ == '__main__':
    (xs, ys) = read_plot()
    plt.plot(xs, ys)
    plt.show()

