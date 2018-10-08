# decorator for registering the survey footprint loaders and projections
projection_register = {}
def register_projection(cls):
    projection_register[cls.__name__] = cls

survey_register = {}
def register_survey(cls):
    survey_register[cls.__name__] = cls

# [blatant copy from six to avoid dependency]
# python 2 and 3 compatible metaclasses
# see http://python-future.org/compatible_idioms.html#metaclasses
def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(type):

        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)

        @classmethod
        def __prepare__(cls, name, this_bases):
            return meta.__prepare__(name, bases)
    return type.__new__(metaclass, 'temporary_class', (), {})

from . import survey
from .map import *
from .projection import *

def loadMap(surveyname, ax=None):
    survey = survey_register[surveyname]
    return Map.load(survey.getConfigfile(), ax=ax)
