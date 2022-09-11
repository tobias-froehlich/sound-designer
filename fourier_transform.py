import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack


def fourier_transform(x, y):
    N = len(x)
    T = x[1] - x[0]
    yf = scipy.fftpack.irfft(y)
    xf = np.linspace(0.0, 1.0/(2*T), N)
    return (xf, yf)




if __name__ == '__main__':

    x_spectrum = np.linspace(0, 1000, 1001) * 2.0
    y_spectrum = np.zeros(1001)
    y_spectrum[10] = 1.

    plt.plot(x_spectrum, y_spectrum)
    plt.show()


    (xf, yf) = fourierTransform(x_spectrum, y_spectrum)
    plt.plot(xf, yf)
    plt.show()
