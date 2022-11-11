import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import matplotlib.image
import threading
from const import *

im = matplotlib.image.imread("bla.png")
im = np.mean(im[:,:,:3], axis=2)
im = np.flipud(im)

print(im.shape)

nFrequencies = int(MAX_FREQUENCY / FREQUENCY_RESOLUTION)
nFrames = int(SAMPLE_RATE * LENGTH_IN_SECONDS)

imageTimes = np.linspace(0.0, LENGTH_IN_SECONDS, int(LENGTH_IN_SECONDS / TIME_RESOLUTION))
print(imageTimes.shape)
print(imageTimes[-1])

times = np.linspace(0.0, LENGTH_IN_SECONDS, int(LENGTH_IN_SECONDS * SAMPLE_RATE))
print(times.shape)
print(times[-1])


def calculateValues(threadIndex):
    values = np.zeros((nFrames,), dtype='float')
    numberOfFrequenciesPerThread = nFrequencies // NUMBER_OF_THREADS + 1
    start = threadIndex * numberOfFrequenciesPerThread
    end = min((threadIndex+1)*numberOfFrequenciesPerThread, nFrequencies)
    for iFrequency in range(start, end):
        print("        "*threadIndex, iFrequency, "of", nFrequencies)
        if im[iFrequency,:].sum() > 0:
            amplitudes = np.interp(times, imageTimes, im[iFrequency,:])
            frequency = iFrequency * FREQUENCY_RESOLUTION
            randomPhase = np.random.random() * 2.0 * np.pi
            values += np.sin(times * frequency * 2.0 * np.pi + randomPhase) * amplitudes * (110.0 / (frequency + 110.0))
        else:
            print("skip")
    valuesCollection[threadIndex] = values


threads = []
valuesCollection = [None,]*NUMBER_OF_THREADS
for threadIndex in range(NUMBER_OF_THREADS):
    threads.append(threading.Thread(target=calculateValues, args=(threadIndex,)))

for threadIndex in range(NUMBER_OF_THREADS):
    threads[threadIndex].start()

for threadIndex in range(NUMBER_OF_THREADS):
    threads[threadIndex].join()


values = sum(valuesCollection)


plt.plot(times, values)
plt.show()

normalized = values / np.max(values)
write('test.wav', SAMPLE_RATE, normalized)
