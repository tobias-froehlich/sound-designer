import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from const import *
dpi = 1024 # Dieser Wert muesste fuer das abgespeicherte Ergebnis egal sein,
         # und nur die Anzeige beeinflussen, wenn man eine mit pl.show() macht.
#         image = pl.imread("structure.png")
py = int(MAX_FREQUENCY)
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

for marker in FREQUENCY_MARKER:
    roundedMarker = round(marker)
    if roundedMarker < py:
        a[round(marker),:,:] = 1.0

matplotlib.image.imsave('bla.png', a, vmin=0.0, vmax=1.0, cmap='Greys', origin='lower')
