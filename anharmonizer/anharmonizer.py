import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.optimize import curve_fit
from const import *

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


class Anharmonizer:
    def __init__(self, filename, batching, cutoffFrequency):
        self.__filename = filename
        self.__batching = batching
        self.__cutoffFrequency = cutoffFrequency
        self.__sampleRate = None
        self.__ts = None
        self.__wav = None
        self.__frequencies = None
        self.__reSpectrumL = None
        self.__imSpectrumL = None
        self.__reSpectrumR = None
        self.__imSepetrumR = None
        self.__peakFrequencies = None

    def readWav(self):
        self.__sampleRate, self.__wav = read(os.path.join("input_data", self.__filename))
        self.__length = (len(self.__wav) - 1) / self.__sampleRate
        self.__ts = np.linspace(0, self.__length, len(self.__wav))
        assert len(self.__wav % 2 == 0)
    
    def writeWav(self):
        write(os.path.join("output_data", self.__filename), self.__sampleRate, np.array(self.__wav, dtype="int16"))

    def transform(self):
        (frequencies, self.__reSpectrumL, self.__imSpectrumL) = self.__transform(self.__ts, self.__wav[:, 0])
        (frequencies, self.__reSpectrumR, self.__imSpectrumR) = self.__transform(self.__ts, self.__wav[:, 1])
        self.__frequencies = frequencies

    def transformBack(self):
        wavL = self.__transformBack(self.__frequencies, self.__reSpectrumL, self.__imSpectrumL)
        wavR = self.__transformBack(self.__frequencies, self.__reSpectrumR, self.__imSpectrumR)
        self.__wav = np.stack([wavL, wavR], 1)

    def findOvertones(self):
        amplitudes = np.sqrt(self.__reSpectrumL**2 + self.__imSpectrumL**2 + self.__reSpectrumR**2 + self.__imSpectrumR**2)
        plt.plot(self.__frequencies, amplitudes, "k")
        maximum =  max(amplitudes)
        indexes_with_high_amplitude = np.array(range(len(amplitudes)))[amplitudes >= maximum * 0.05]
        firstPeakIndex = amplitudes[:int(indexes_with_high_amplitude[0] * 1.5)].argmax()
        peakFrequencies = [self.__findPeak(self.__frequencies[firstPeakIndex], amplitudes)]
        for i in range(2, 32 + 1):
            if i == 2:
                expectedFrequency = 2 * peakFrequencies[-1]
            else:
                expectedFrequency = peakFrequencies[-2] + 2 * (peakFrequencies[-1] - peakFrequencies[-2])
            exactFrequency = self.__findPeak(expectedFrequency, amplitudes)
            if exactFrequency == None:
                peakFrequencies.append(expectedFrequency)
            else:
                peakFrequencies.append(exactFrequency)
        for frequency in peakFrequencies:
            plt.axvline(frequency, alpha=0.2)
        plt.xlim([0, 5 * peakFrequencies[0]])
        plt.savefig(os.path.join("output_data", self.__filename + ".png"))
        plt.close()
        self.__peakFrequencies = peakFrequencies
        
        
        

    def __findPeak(self, frequency, amplitudes):
        indexes = range(len(self.__frequencies))
        leftIndex = int(np.interp(frequency - 30, self.__frequencies, indexes))
        rightIndex = int(np.interp(frequency + 30, self.__frequencies, indexes))
        xs = self.__frequencies[leftIndex:rightIndex]
        ys = amplitudes[leftIndex:rightIndex]
        if len(xs) == 0:
            return None
        A0 = ys.max()
        mu0 = xs[ys.argmax()]
        sigma0 = 0.5
        try:
            popt, pcov = curve_fit(gauss, xs, ys, p0=[A0, mu0, sigma0])
            A = popt[0]
            mu = popt[1]
            sigma = popt[2]
        except:
            mu = None

        if mu == None:
            return None
        elif mu < frequency - 30 or mu > frequency + 30 or sigma > 5.0:
            return None
        else:
            return mu
        


    def manipulateSpectrum(self):
        indexes = range(len(self.__reSpectrumL))
        self.__targetFrequencies = []
        baseFrequency = self.__peakFrequencies[0]
        inverseExponent = 1.0 / STRETCH_EXPONENT
        def f(frequency):
            return baseFrequency * (frequency / baseFrequency)**inverseExponent
        reL = []
        imL = []
        reR = []
        imR = []
        
#        for i in indexes:
#            reL.append(np.interp(f(self.__frequencies[i]), self.__frequencies, self.__reSpectrumL))
#            imL.append(np.interp(f(self.__frequencies[i]), self.__frequencies, self.__imSpectrumL))
#            reR.append(np.interp(f(self.__frequencies[i]), self.__frequencies, self.__reSpectrumR))
#            imR.append(np.interp(f(self.__frequencies[i]), self.__frequencies, self.__imSpectrumR))
        frequenciesToUse = self.__frequencies[self.__frequencies < self.__sampleRate]
        missingLength = len(self.__frequencies) - len(frequenciesToUse)
        reL = np.interp(f(self.__frequencies), self.__frequencies, self.__reSpectrumL)
        imL = np.interp(f(self.__frequencies), self.__frequencies, self.__imSpectrumL)
        reR = np.interp(f(self.__frequencies), self.__frequencies, self.__reSpectrumR)
        imR = np.interp(f(self.__frequencies), self.__frequencies, self.__imSpectrumR)
        
        zeroArray = np.zeros((max(0, missingLength),))
        reL = np.concatenate([reL, zeroArray])
        imL = np.concatenate([imL, zeroArray])
        reR = np.concatenate([reR, zeroArray])
        imR = np.concatenate([imR, zeroArray])
        
        self.__reSpectrumL = reL
        self.__imSpectrumL = imL
        self.__reSpectrumR = reR
        self.__imSpectrumR = imR

    def __transform(self, ts, amplitudes):
        frequencies = np.linspace(0, self.__sampleRate // 2, int(self.__length * self.__sampleRate // 2))
        transformed = np.fft.fft(amplitudes) / self.__sampleRate
        transformed = transformed[:int(self.__length * self.__sampleRate // 2)]
        re = transformed.real
        im = transformed.imag
        return (frequencies, re, im)
        

    def __transformBack(self, frequencies, re, im):
        re_full = np.concatenate([re, np.zeros(len(re))])
        im_full = np.concatenate([im, np.zeros(len(im))])

        return np.fft.fft(re_full - 1j * im_full).real / self.__length * 2.0



if __name__ == "__main__":
    filenames = os.listdir("input_data")
    for filename in filenames:
        print(filename)
        anharmonizer = Anharmonizer(filename, 10, 20000.0)
        anharmonizer.readWav()
        anharmonizer.transform()
        anharmonizer.findOvertones()
        anharmonizer.manipulateSpectrum()
        anharmonizer.transformBack()
        anharmonizer.writeWav()
