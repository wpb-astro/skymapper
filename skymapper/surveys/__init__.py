import os

# decorated function to load footprint ra,dec from file
# need one of those per survey

from .. import skymapper
from ..skymapper import register, np

@register(surveyname="DES")
def DESFP():
    """Returns RA, Dec of the survey footprint."""
    this_dir, this_filename = os.path.split(__file__)
    datafile = os.path.join(this_dir, "des-round17-poly.txt")
    data = np.loadtxt(datafile)
    return data[:,0], data[:,1]
