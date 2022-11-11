import numpy as np
from scipy.io.wavfile import write
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import matplotlib.image
import threading
from const import *


rate, values = read('eulerbell_1.99_3.02_4.97_7.15_reverb.wav')

print(values.shape)
for i in range(6):
    if i > 0:
        if values.shape[0] % 2 == 1:
            values = values[:-1];
        values = values.reshape((values.shape[0]//2, 2)).mean(axis=1)
    print('test_%i.wav'%(i))
    write('test_%i.wav'%(i), 96000, values)
