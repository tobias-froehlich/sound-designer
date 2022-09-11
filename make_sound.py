import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import matplotlib.image
from const import *

im = matplotlib.image.imread("bla.png")
im = np.mean(im[:,:,:3], axis=2)
im = np.flipud(im)

print(im.shape)

nFrequencies = int(MAX_FREQUENCY / FREQUENCY_RESOLUTION)
nFrames = int(SAMPLE_FREQUENCY * LENGTH_IN_SECONDS)

imageTimes = np.linspace(0.0, LENGTH_IN_SECONDS, int(LENGTH_IN_SECONDS / TIME_RESOLUTION))
print(imageTimes.shape)
print(imageTimes[-1])

times = np.linspace(0.0, LENGTH_IN_SECONDS, int(LENGTH_IN_SECONDS * SAMPLE_FREQUENCY))
print(times.shape)
print(times[-1])

values = np.zeros((nFrames,), dtype='float')
for iFrequency in range(nFrequencies):
    print(iFrequency, "of", nFrequencies)
    amplitudes = np.interp(times, imageTimes, im[iFrequency,:])
    frequency = iFrequency * FREQUENCY_RESOLUTION
    randomPhase = np.random.random() * 2.0 * np.pi
    values += np.sin(times * frequency * 2.0 * np.pi + randomPhase) * amplitudes
    
plt.plot(times, values)
plt.show()

normalized = values / np.max(values)
write('test.wav', 44100, normalized)
