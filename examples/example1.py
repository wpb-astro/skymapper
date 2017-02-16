# load projection and helper functions
import numpy as np
import skymapper as skm
import matplotlib.pylab as plt

def getCatalog(size=10000):
    # dummy catalog
    ra = np.random.uniform(size=size, low=-55, high=100)
    dec = np.random.uniform(size=size, low=-65, high=0)
    return ra, dec

if __name__ == "__main__":

    # load RA/Dec from catalog
    size = 100000
    ra, dec = getCatalog(size)

    # plot density in healpix cells
    nside = 64
    sep = 15

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111, aspect='equal')
    _, _, proj = skm.plotDensity(ra, dec, nside=nside, sep=sep, ax=ax)

    # add DES footprint
    skm.plotFootprint('DES', proj, ax=ax, zorder=10, edgecolor='#2222B2', facecolor='None', lw=2)

    # add title
    #fig.suptitle('Silly random in DES')
