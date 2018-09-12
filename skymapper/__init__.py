# decorator for registering the survey footprint loaders and projections
projection_register = {}
def register_projection(cls):
    projection_register[cls.__name__] = cls

survey_register = {}
def register_survey(cls):
    survey_register[cls.__name__] = cls

from . import survey
from .map import *
from .projection import *

def loadMap(surveyname, ax=None):
    survey = survey_register[surveyname]
    return Map.load(survey.getConfigfile(), ax=ax)


def getOptimalConicProjection(ra, dec, proj_class=None, ra0=None, dec0=None):
    """Determine optimal configuration of conic map.

    As a simple recommendation, the standard parallels are chosen to be 1/7th
    closer to dec0 than the minimum and maximum declination in the data
    (Snyder 1987, page 99).

    If proj_class is None, it will use AlbersEqualAreaProjection.

    Args:
        ra: list of rectascensions
        dec: list of declinations
        proj_class: constructor of projection class
        ra0: if not None, use this as reference RA
        dec0: if not None, use this as reference Dec

    Returns:
        proj_class that best holds ra/dec
    """

    if ra0 is None:
        ra_ = np.array(ra)
        ra_[ra_ > 180] -= 360
        ra_[ra_ < -180] += 360
        # weight more towards the poles because that decreases distortions
        ra0 = (ra_ * dec).sum() / dec.sum()

    if dec0 is None:
        dec0 = np.median(dec)

    # determine standard parallels for AEA
    dec1, dec2 = dec.min(), dec.max()
    # move standard parallels 1/6 further in from the extremes
    # to minimize scale variations (Snyder 1987, section 14)
    delta_dec = (dec0 - dec1, dec2 - dec0)
    dec1 += delta_dec[0]/7
    dec2 -= delta_dec[1]/7

    if proj_class is None:
        proj_class = AlbersEqualAreaProjection
    return proj_class(ra0, dec0, dec1, dec2)
