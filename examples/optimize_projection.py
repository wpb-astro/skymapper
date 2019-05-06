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

if __name__ == "__main__":

    # load RA/Dec from catalog
    size = 10000
    des = skm.survey.DES()
    ra, dec = getCatalog(size, survey=des)

    # define the best WagnerIV projection for the footprint
    # minimizing the variation in distortion, aka ellipticity
    for crit in [skm.meanDistortion, skm.maxDistortion, skm.stdDistortion]:
        proj = skm.WagnerIV.optimize(ra, dec, crit)
        map = skm.Map(proj)
        map.grid()
        #map.labelMeridianAtParallel(-90, meridians=[])
        map.footprint(des, nside=64, zorder=20, facecolor='w', alpha=0.3)
        a,b = proj.distortion(ra, dec)
        c = map.extrapolate(ra, dec, 1-np.abs(b/a), vmin=0, vmax=0.3, resolution=72)
        cb = map.colorbar(c, cb_label='distortion')
        map.focus(ra, dec)
        map.title(proj.__class__.__name__ + ": " + crit.__name__)
