# load projection and helper functions
import numpy as np
import skymapper as skm
import matplotlib.pylab as plt

def getCatalog(size=10000):
    # dummy catalog
    ra = np.random.uniform(size=size, low=-55, high=100)
    dec = np.random.uniform(size=size, low=-65, high=0)
    return ra, dec

def makeHealpixMap(ra, dec, nside=1024, nest=False):
    # convert a ra/dec catalog into healpix map with counts per cell
    import healpy as hp
    ipix = hp.ang2pix(nside, (90-dec)/180*np.pi, ra/180*np.pi, nest=nest)
    return np.bincount(ipix, minlength=hp.nside2npix(nside))

def getHealpixCoords(pixels, nside, nest=False):
    # convert healpix cell indices to center ra/dec
    import healpy as hp
    theta, phi = hp.pix2ang(nside, pixels, nest=nest)
    return phi * 180. / np.pi, 90 - theta * 180. / np.pi


if __name__ == "__main__":

    # load RA/Dec from catalog
    size = 100000
    ra, dec = getCatalog(size)

    # plot density in healpix cells
    nside = 64
    sep = 15

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111, aspect='equal')
    fig, ax, proj = skm.plotDensity(ra, dec, nside=nside, sep=sep, ax=ax)

    # add DES footprint
    skm.addFootprint('DES', proj, ax, zorder=10, edgecolor='#2222B2', facecolor='None', lw=2)

    # test Healpix map functions
    try:
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111, aspect='equal')

        m = makeHealpixMap(ra, dec, nside=nside)
        fig, ax, proj = skm.plotHealpix(m, nside, sep=sep, ax=ax, cb_label="Healpix cell count")
        skm.addFootprint('DES', proj, ax, zorder=10, edgecolor='#2222B2', facecolor='None', lw=2)

        # make free-form map (only works for equal-area projections)
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(111, aspect='equal')

        pixels = np.flatnonzero(m)
        ra_, dec_ = getHealpixCoords(pixels, nside)
        fig, ax, proj = skm.plotMap(ra_, dec_, m[pixels], sep=sep, ax=ax, cmap='YlOrRd', cb_label="Map value")
        skm.addFootprint('DES', proj, ax, zorder=10, edgecolor='#2222B2', facecolor='None', lw=2)

    except ImportError:
        pass
