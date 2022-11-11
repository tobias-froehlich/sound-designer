import numpy as np
from scipy.io.wavfile import write
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import matplotlib.image
import threading
from const import *


rate, values = read('test.wav')
print(values)
write('test2.wav', 96000, values)
