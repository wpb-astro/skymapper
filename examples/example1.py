# load projection and helper functions
import numpy as np
import skymapper as skm
import matplotlib.pylab as plt

def getCatalog(size=10000, survey=None):
    # dummy catalog: uniform on sphere
    # Marsaglia (1972)
    xyz = np.random.normal(size=(size, 3))
    r = np.sqrt((xyz**2).sum(axis=1))
    dec = np.arccos(xyz[:,2]/r) / skm.DEG2RAD - 90
    ra = - np.arctan2(xyz[:,0], xyz[:,1]) / skm.DEG2RAD

    if survey is not None:
        inside = survey.contains(ra, dec)
        ra = ra[inside]
        dec = dec[inside]

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
    des = skm.survey.DES()
    ra, dec = getCatalog(size, survey=des)

    # define the best Albers projection for the footprint
    # minimizing the variation in distortion
    crit = skm.stdDistortion
    proj = skm.Albers.optimize(ra, dec, crit=crit)

    # construct map: will hold figure and projection
    # the outline of the sphere can be styled with kwargs for matplotlib Polygon
    map = skm.Map(proj)

    # add graticules, separated by 15 deg
    # the lines can be styled with kwargs for matplotlib Line2D
    # additional arguments for formatting the graticule labels
    sep=15
    map.grid(sep=sep)

    # # add footprint, retain the polygon for clipping
    # footprint = map.footprint("DES", zorder=20, edgecolor='#2222B2', facecolor='None', lw=1)
    #
    #### 1. plot density in healpix cells ####
    nside = 32
    mappable = map.density(ra, dec, nside=nside)
    cb = map.colorbar(mappable, cb_label="$n$ [arcmin$^{-2}$]")

    # add random scatter plot
    len = 10
    size = 100*np.random.rand(len)
    map.scatter(ra[:len], dec[:len], s=size, edgecolor='k', facecolor='None')

    # focus on relevant region
    map.focus(ra, dec)

    # entitle: access mpl figure
    map.fig.suptitle('Density with random scatter')

    # copy map without data contents
    map2 = map.clone()

    #### 2. show map distortion over the survey ####
    a,b = proj.distortion(ra, dec)
    mappable2 = map2.bin(ra, dec, 1-np.abs(b/a), 32, vmin=0, vmax=0.3, cmap='RdYlBu_r')
    cb2 = map2.colorbar(mappable2, cb_label='Distortion')
    map2.fig.suptitle('Projection distortion')

    #### 3. extrapolate RA over all sky ####
    map3 = skm.Map(proj)

    # show with 45 deg graticules
    sep=45
    map3.grid(sep=sep)

    # alter number of labels at the south pole
    map3.labelMeridiansAtParallel(-90, size=8, meridians=np.arange(0,360,90))

    # this is slow when working with lots of samples...
    mappable3 = map3.extrapolate(ra[::10], dec[::10], dec[::10], nside=nside)
    cb3 = map3.colorbar(mappable3, cb_label='Dec')

    # add footprint shade
    footprint3 = map3.footprint(des, nside=nside, zorder=20, facecolors='k', alpha=0.1)

    map3.fig.suptitle('Extrapolation on the sphere')

    #### 4. test Healpix map functions ####
    map4 = map.clone()

    # simply bin the counts of ra/dec
    m = makeHealpixMap(ra, dec, nside=nside)
    mappable4 = map4.healpix(m, cmap="YlOrRd")
    cb4 = map4.colorbar(mappable4, cb_label="Healpix cell count")
    map4.fig.suptitle('Healpix map')
