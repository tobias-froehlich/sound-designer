from fourier_transform import fourier_transform
from read_plot import read_plot
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write
from const import *

(xs, ys) = read_plot()
(xs, ys) = fourier_transform(xs, ys)
plt.plot(xs, ys)
plt.show()

normalized = ys[int(44100*0.2):int(44100*(ELONGATION_FACTOR/2 - 0.2))]
normalized = normalized / np.max(normalized)
write('test.wav', 44100, normalized)
