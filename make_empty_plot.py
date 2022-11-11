import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from primeFactors import getFactors
from const import *
dpi = 1024 # Dieser Wert muesste fuer das abgespeicherte Ergebnis egal sein,
         # und nur die Anzeige beeinflussen, wenn man eine mit pl.show() macht.
#         image = pl.imread("structure.png")
py = int(MAX_FREQUENCY / FREQUENCY_RESOLUTION)
px = int(LENGTH_IN_SECONDS / TIME_RESOLUTION)
#fig = plt.figure(figsize=(px/np.float(dpi), py/np.float(dpi)), dpi=dpi)
#ax = fig.add_axes([0, 0, 1, 1])
#ax.set_xticks(FREQUENCY_MARKER)
#ax.set_yticks([])
#ax.tick_params(direction='in', length=6, pad=-15)
#ax.set_xlim([0, MAX_FREQUENCY+1])
#ax.spines['left'].set_color('white')
#ax.spines['top'].set_color('white')
#plt.axhline(y=0.2, color='black', linewidth=0.5)
##ax.axis("off")
##         ax.imshow(np.flipud(image), extent=[0, 1, 0, 1], origin="lower")
#plt.savefig("spectrum.png")
#plt.close()

a = np.zeros((py, px, 3), 'float32')

frequency_markers = []
for n in range(1, 1024):
    factors = getFactors(n)
    print(n, factors)
    marker = 110.0
    for factor in factors:
        if factor == 2:
            marker *= FACTOR_TWO
        elif factor == 3:
            marker *= FACTOR_THREE
        elif factor == 5:
            marker *= FACTOR_FIVE
        elif factor == 7:
            marker *= FACTOR_SEVEN
        else:
            marker *= 0
    if marker > 0:
        frequency_markers.append(marker)

for marker in frequency_markers:
    roundedMarker = round(marker/FREQUENCY_RESOLUTION)
    if roundedMarker < py:
        a[roundedMarker,:,:] = 1.0

matplotlib.image.imsave('bla.png', a, vmin=0.0, vmax=1.0, cmap='Greys', origin='lower')
