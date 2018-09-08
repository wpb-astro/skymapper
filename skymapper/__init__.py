# decorator for registering the survey footprint loaders and projections
projection_register = {}
def register_projection(cls):
    projection_register[cls.__name__] = cls

survey_register = {}
def register_survey(cls):
    survey_register[cls.__name__] = cls

from . import survey
from .skymapper import *
from .projection import *
