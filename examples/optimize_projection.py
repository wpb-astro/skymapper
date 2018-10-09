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

if __name__ == "__main__":

    # load RA/Dec from catalog
    size = 10000
    ra, dec = getCatalog(size, surveyname="DES")

    # define the best WagnerIV projection for the footprint
    # minimizing the variation in distortion, aka ellipticity
    for crit in [skm.meanDistortion, skm.maxDistortion, skm.stdDistortion]:
        proj = skm.WagnerIV.optimize(ra, dec, crit)
        map = skm.Map(proj)
        map.grid()
        map.labelMeridianAtParallel(-90, meridians=[])
        footprint = map.footprint("DES", zorder=20, edgecolor='#2222B2', facecolor='None', lw=1)
        a,b = proj.distortion(ra, dec)
        c = map.extrapolate(ra, dec, 1-np.abs(b/a), vmin=0, vmax=0.3, resolution=72, clip_path=footprint)
        cb = map.colorbar(c, cb_label='distortion')
        map.focus(ra, dec)
        map.fig.suptitle(crit.__name__)
