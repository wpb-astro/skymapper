# load projection and helper functions
import numpy as np
import skymapper as skm
import matplotlib.pylab as plt

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __getitem__(self, i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        raise NotImplementedError

def getCatalog(size=10000, surveyname=None):
    # dummy catalog: uniform on sphere
    # Marsaglia (1972)
    xyz = np.random.normal(size=(size, 3))
    r = np.sqrt((xyz**2).sum(axis=1))
    dec = np.arccos(xyz[:,2]/r) / skm.DEG2RAD - 90
    ra = - np.arctan2(xyz[:,0], xyz[:,1]) / skm.DEG2RAD

    if surveyname is not None:
        from matplotlib.patches import Polygon
        # construct survey polygon
        ra_fp, dec_fp = skm.survey_register[surveyname].getFootprint()
        poly = Polygon(np.dstack((ra_fp,dec_fp))[0], closed=True)
        inside = [poly.get_path().contains_point(Point(ra_,dec_)) for (ra_,dec_) in zip(ra,dec)]
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
    ra, dec = getCatalog(size, surveyname="DES")

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

    # alter position of default labels at the outer meridians
    for m in [proj.ra_0 + 180, proj.ra_0 - 180]:
        map.labelParallelAtMeridian(m, verticalalignment='top', horizontalalignment='center')

    # remove labels at the south pole
    map.labelMeridianAtParallel(-90, meridians=[])

    # add footprint, retain the polygon for clipping
    footprint = map.footprint("DES", zorder=20, edgecolor='#2222B2', facecolor='None', lw=1)

    #### 1. plot density in healpix cells ####
    nside = 32
    mappable = map.density(ra, dec, nside=nside, clip_path=footprint)
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
    footprint2 = map2.footprint("DES", zorder=20, edgecolor='#2222B2', facecolor='None', lw=1)

    #### 2. show map distortion over the survey ####
    a,b = proj.distortion(ra, dec)
    mappable2 = map2.interpolate(ra, dec, 1-np.abs(b/a), vmin=0, vmax=0.3, clip_path=footprint2)
    cb2 = map2.colorbar(mappable2, cb_label='Distortion')
    map2.fig.suptitle('Projection distortion')

    #### 3. extrapolate RA over all sky ####
    map3 = skm.Map(proj)

    # show with 45 deg graticules
    sep=45
    map3.grid(sep=sep)

    # alter position of default labels at the outer meridians
    for m in [proj.ra_0 + 180, proj.ra_0 - 180]:
        map3.labelParallelAtMeridian(m, verticalalignment='top', horizontalalignment='center')

    # alter number of labels at the south pole
    map3.labelMeridianAtParallel(-90, size=8, meridians=np.arange(0,360,90))

    footprint3 = map3.footprint("DES", zorder=20, edgecolor='#2222B2', facecolor='None', lw=1)
    # this is slow when working with lots of samples...
    mappable3 = map3.extrapolate(ra[::10], dec[::10], dec[::10])
    cb3 = map3.colorbar(mappable3, cb_label='Dec')
    map3.fig.suptitle('Extrapolation on the sphere')

    #### 4. test Healpix map functions ####
    try:
        # simply bin the counts of ra/dec
        m = makeHealpixMap(ra, dec, nside=nside)

        map4 = map.clone()
        footprint4 = map4.footprint("DES", zorder=20, edgecolor='#2222B2', facecolor='None', lw=1)

        mappable4 = map4.healpix(m, clip_path=footprint4, cmap="YlOrRd")
        cb4 = map4.colorbar(mappable2, cb_label="Healpix cell count")
        map4.fig.suptitle('Healpix map')

    except ImportError:
        pass
