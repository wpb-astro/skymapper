import numpy as np
import scipy.integrate
import scipy.optimize

DEG2RAD = np.pi/180

def _toArray(x):
    """Convert x to array if needed

    Returns:
        array(x), boolean if x was an array before
    """
    if hasattr(x, '__iter__'):
        return np.array(x), True
    return np.array([x], dtype=np.double), False

def ellipticity(a, b):
    """Returns 1-abs(b/a)"""
    return 1-np.abs(b/a)
def meanDistortion(a, b):
    """Returns average `ellipticity` over all `a`,`b`"""
    return np.mean(ellipticity(a,b))
def maxDistortion(a,b):
    """Returns max `ellipticity` over all `a`,`b`"""
    return np.max(ellipticity(a,b))
def stdDistortion(a,b):
    """Returns `std(b/a)`"""
    return (b/a).std() # include the sign
def stdScale(a,b):
    """Returns `std(a*b)`

    This is useful for conformal projections.
    """
    return (a*b).std()
def stdDistortionScale(a,b):
    """Retruns sum of `stdScale` and `stdDistortion`.

    This is useful for a compromise between equal-area and conformal projections.
    """
    return stdScale(a,b) + stdDistortion(a,b)

def _optimize_objective(x, proj_cls, lon_type, lon, lat, crit):
    """Construct projections from parameters `x` and compute `crit` for `lon, lat`"""
    proj = proj_cls(*x, lon_type=lon_type)
    a, b = proj.distortion(lon, lat)
    return crit(a,b)

def _optimize(proj_cls, x0, lon_type, lon, lat, crit, bounds=None):
    """Determine parameters for `proj_cls` that minimize `crit` over `lon, lat`.

    Args:
        proj_cls: projection class
        x0: arguments for projection class `__init__`
        lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)
        lon: list of rectascensions
        lat: list of declinations
        crit: optimization criterion
            needs to be function of semi-major and semi-minor axes of the Tissot indicatix
        bounds: list of upper and lower bounds on each parameter in `x0`

    Returns:
        optimized projection of class `proj_cls`
    """
    print ("optimizing parameters of %s to minimize %s" % (proj_cls.__name__, crit.__name__))
    x, fmin, d = scipy.optimize.fmin_l_bfgs_b(_optimize_objective, x0, args=(proj_cls, lon_type, lon, lat, crit), bounds=bounds, approx_grad=True)
    res = proj_cls(*x, lon_type=lon_type)
    print ("best objective %.6f at %r" % (fmin, res))
    return res

def _dist(radec, proj, xy):
    return np.sum((xy - np.array(proj(radec[0], radec[1])))**2)


class BaseProjection(object):
    """Projection base class

    Every projection needs to implement three methods:
    * `transform(self, lon, lat)`: mapping from lon/lat to map x/y
    * `invert(self, x, y)`: the inverse mapping from x/y to lon/lat

    All methods accept either single number or arrays and return accordingly.
    """
    def __init__(self, lon_0=0, lon_type="ra"):
        """Initialize projection

        Args:
            lon_0 (int, float):  reference longitude
            lon_type (string): type of longitude
                "lon" for a standard coordinate system (right-handed, -180..180 deg)
                "ra" for n equatorial coordinate system (left-handed, 0..360 deg)
        """
        assert lon_type in ['ra', 'lon']
        self.lon_0 = lon_0
        self.lon_type = lon_type
        if self.lon_type == "ra" and self.lon_0 < 0:
            self.lon_0 += 360
        elif self.lon_type == "lon" and self.lon_0 > 180:
            self.lon_0 -= 360

    def __call__(self, lon, lat):
        """Shorthand for `transform`. Works also with single coordinates

        Args:
            lon (float, array): longitude
            lat (float, array): latitude
        """
        lon_, isArray = _toArray(lon)
        lat_, isArray = _toArray(lat)
        assert len(lon_) == len(lat_)

        x, y = self.transform(lon_, lat_)
        if isArray:
            return x, y
        else:
            return x[0], y[0]

    def transform(self, lon, lat):
        """Convert longitude/latitude into map coordinates

        Note: Unlike `__call__`, lon/lat need to be arrays!

        Args:
            lon (array): longitudes
            lat (array): latitudes

        Returns:
            x,y with the same format as lon,lat
        """
        pass

    def inv(self, x, y):
        """Shorthand for `invert`. Works also with single coordinates

        Args:
            x (float, array): horizontal map coordinate
            y (float, array): vertical map coordinate
        """
        x_, isArray = _toArray(x)
        y_, isArray = _toArray(y)
        lon, lat = self.invert(x_, y_)
        if isArray:
            return lon, lat
        else:
            return lon[0], lat[0]

    def invert(self, x, y):
        """Convert map coordinates into longitude/latitude

        Args:
            x (array): horizontal map coordinates
            y (array): vertical map coordinates

        Returns:
            lon,lat with the same format as x,y
        """
        # default implementation for non-analytic inverses
        assert len(x) == len(y)

        bounds = ((None,None), (-90, 90)) # lon/lat limits
        start = (self.lon_0,0) # lon/lat of initial guess: should be close to map center
        lon, lat = np.empty(len(x)), np.empty(len(y))
        i = 0
        for x_,y_ in zip(x, y):
            xy = np.array([x_,y_])
            radec, fmin, d = scipy.optimize.fmin_l_bfgs_b(_dist, start, args=(self, xy), bounds=bounds, approx_grad=True)
            if fmin < 1e-6: # smaller than default tolerance of fmin
                lon[i], lat[i] = radec
            else:
                lon[i], lat[i] = -1000, -1000
            i += 1

        return lon, lat

    @property
    def poleIsPoint(self):
        """Whether the pole is mapped onto a point"""
        try:
            return self._poleIsPoint
        except AttributeError:
            self._poleIsPoint = {}
            N = 10
            # run along the poles from the left to right outer meridian
            rnd_meridian = -180 + 360*np.random.rand(N) + self.lon_0
            for deg in [-90, 90]:
                line = self.transform(rnd_meridian, deg*np.ones(N))
                if np.unique(line[0]).size > 1 and np.unique(line[1]).size > 1:
                    self._poleIsPoint[deg] = False
                else:
                    self._poleIsPoint[deg] = True
            return self._poleIsPoint

    def _standardize(self, lon):
        """Normalize longitude to -180 .. 180, with reference `lon_0` at 0"""
        lon_ = lon - self.lon_0 # need copy to prevent changing data
        if self.lon_type == "ra":
            lon_ *= -1 # left-handed
        # check that lon_ is between -180 and 180 deg
        lon_[lon_ < -180 ] += 360
        lon_[lon_ > 180 ] -= 360
        return lon_

    def _unstandardize(self, lon):
        """Revert `_standardize`"""
        # no copy needed since all lons have been altered/transformed before
        if self.lon_type == "ra":
            lon *= -1 # left-handed
        lon += self.lon_0
        lon [lon < 0] += 360
        lon [lon > 360] -= 360
        return lon

    def gradient(self, lon, lat, sep=1e-2, direction='parallel'):
        """Compute the gradient in map coordinates at given sky position

        Note: Gradient along parallel is computed in positive lon direction

        Args:
            lon: (list of) longitude
            lat: (list of) latitude
            sep: distance for symmetric first-order derivatives
            direction: tangent direction for gradient, from `['parallel', 'meridian']`

        Returns:
            `dx`, `dy` for every item in `lon/lat`
        """
        assert direction in ['parallel', 'meridian']

        lon_, isArray = _toArray(lon)
        lat_, isArray = _toArray(lat)

        # gradients in *positive* lat and *negative* lon
        if direction == 'parallel':
            test = np.empty((2, lon_.size))
            test[0] = lon_-sep/2
            test[1] = lon_+sep/2

            # check for points beyond -180 / 180
            mask = test[0] <= self.lon_0 - 180
            test[0][mask] = lon_[mask]
            mask = test[1] >= self.lon_0 + 180
            test[1][mask] = lon_[mask]

            x, y = self.transform(test, np.ones((2,lon_.size))*lat)
        else:
            test = np.empty((2, lat_.size))
            test[0] = lat_-sep/2
            test[1] = lat_+sep/2

            # check for points beyond -90 / 90
            mask = test[0] <= -90
            test[0][mask] = lat_[mask]
            mask = test[1] >= 90
            test[1][mask] = lat_[mask]

            x, y = self.transform(np.ones((2,lat_.size))*lon, test)

        sep = test[1] - test[0]
        x[0] = (x[1] - x[0])/sep # dx
        x[1] = (y[1] - y[0])/sep # dy
        if isArray:
            return x.T
        return x[:,0]

    def jacobian(self, lon, lat, sep=1e-2):
        """Jacobian of mapping from lon/lat to map coordinates x/y

        Args:
            lon: (list of) longitude
            lat: (list of) latitude

        Returns:
            ((dx/dlon, dx/dlat), (dy/dlon, dy/dlat)) for every item in `lon/lat`
        """
        dxy_dra= self.gradient(lon, lat, sep=sep, direction='parallel')
        dxy_ddec = self.gradient(lon, lat, sep=sep, direction='meridian')
        return np.dstack((dxy_dra, dxy_ddec))

    def distortion(self, lon, lat):
        """Compute semi-major and semi-minor axis according to Tissot's indicatrix

        See Snyder (1987, section 4)

        Args:
            lon: (list of) longitude
            lat: (list of) latitude

        Returns:
            a, b for every item in `lon/lat`
        """
        jac = self.jacobian(lon,lat)
        cos_phi = np.cos(lat * DEG2RAD)
        h = np.sqrt(jac[:,0,1]**2 + jac[:,1,1]**2)
        k = np.sqrt(jac[:,0,0]**2 + jac[:,1,0]**2) / cos_phi
        sin_t = (jac[:,1,1]*jac[:,0,0] - jac[:,0,1]*jac[:,1,0])/(h*k*cos_phi)
        a_ = np.sqrt(np.maximum(h*h + k*k + 2*h*k*sin_t, 0)) # can be very close to 0
        b_ = np.sqrt(np.maximum(h*h + k*k - 2*h*k*sin_t, 0))
        a = (a_ + b_) / 2
        b = (a_ - b_) / 2
        s = h*k*sin_t
        return a, b

    @classmethod
    def optimize(cls, lon, lat, crit=meanDistortion, lon_type="ra"):
        """Optimize the parameters of projection to minimize `crit` over `lon,lat`

        Args:
            lon: list of longitude
            lat: list of latitude
            crit: optimization criterion
                needs to be function of semi-major and semi-minor axes of the Tissot indicatix
            lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)

        Returns:
            optimized projection
        """
        lon_ = np.array(lon)
        # go into standard frame, right or left-handed is irrelevant here
        lon_[lon_ > 180] -= 360
        lon_[lon_ < -180] += 360
        bounds = ((-180,180),)
        x0 = np.array((lon_.mean(),))
        return _optimize(cls, x0, lon_type, lon, lat, crit, bounds=bounds)

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.lon_0)


# metaclass for registration.
# see https://effectivepython.com/2015/02/02/register-class-existence-with-metaclasses/

from . import register_projection, with_metaclass
# [blatant copy from six to avoid dependency]
# python 2 and 3 compatible metaclasses
# see http://python-future.org/compatible_idioms.html#metaclasses

class Meta(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)

        # remove those that are directly derived from BaseProjection
        if BaseProjection not in bases:
            register_projection(cls)

        return cls

class Projection(with_metaclass(Meta, BaseProjection)):
    pass


class ConicProjection(BaseProjection):
    def __init__(self, lon_0, lat_0, lat_1, lat_2, lon_type="ra"):
        """Base class for conic projections

        Args:
            lon_0: longitude that maps onto x = 0
            lat_0: latitude that maps onto y = 0
            lat_1: lower standard parallel
            lat_2: upper standard parallel (must not be -lat_1)
            lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)
        """
        super(ConicProjection, self).__init__(lon_0, lon_type)
        self.lat_0 = lat_0
        self.lat_1 = lat_1
        self.lat_2 = lat_2
        if lat_1 > lat_2:
            self.lat_1, self.lat_2 = self.lat_2, self.lat_1

    @classmethod
    def optimize(cls, lon, lat, crit=meanDistortion, lon_type="ra"):
        """Optimize the parameters of projection to minimize `crit` over `lon,lat`

        Uses median latitude and latitude-weighted longitude as reference,
        and places standard parallels 1/6 inwards from the min/max latitude
        to minimize scale variations (Snyder 1987, section 14).

        Args:
            lon: list of longitude
            lat: list of latitude
            crit: optimization criterion
                needs to be function of semi-major and semi-minor axes of the Tissot indicatix
            lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)

        Returns:
            optimized projection
        """
        # for conics: need to determine central lon, lat plus two standard parallels
        # normalize lon
        lon_ = np.array(lon)
        lon_[lon_ > 180] -= 360
        lon_[lon_ < -180] += 360
        # weigh more towards the poles because that decreases distortions
        lon0 = (lon_ * lat).sum() / lat.sum()
        if lon0 < 0:
            lon0 += 360
        lat0 = np.median(lat)

        # determine standard parallels
        lat1, lat2 = lat.min(), lat.max()
        delta_lat = (lat0 - lat1, lat2 - lat0)
        lat1 += delta_lat[0]/6
        lat2 -= delta_lat[1]/6

        x0 = np.array((lon0, lat0, lat1, lat2))
        bounds = ((0, 360), (-90,90),(-90,90), (-90,90))
        return _optimize(cls, x0, lon_type, lon, lat, crit, bounds=bounds)

    def __repr__(self):
        return "%s(%r,%r,%r,%r)" % (self.__class__.__name__, self.lon_0, self.lat_0, self.lat_1, self.lat_2)


class Albers(ConicProjection, Projection):
    """Albers Equal-Area conic projection

    AEA is a conic projection with an origin along the lines connecting
    the poles. It preserves relative area, but is not conformal,
    perspective or equistant.

    Its preferred use of for areas with predominant east-west extent
    at moderate latitudes.

    As a conic projection, it depends on two standard parallels, i.e.
    intersections of the cone with the sphere. To minimize scale variations,
    these standard parallels should be chosen as small as possible while
    spanning the range in declinations of the data.

    For details, see Snyder (1987, section 14).
    """
    def __init__(self, lon_0, lat_0, lat_1, lat_2, lon_type="ra"):
        """Create Albers projection

        Args:
            lon_0: longitude that maps onto x = 0
            lat_0: latitude that maps onto y = 0
            lat_1: lower standard parallel
            lat_2: upper standard parallel (must not be -lat_1)
            lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)
        """
        super(Albers, self).__init__(lon_0, lat_0, lat_1, lat_2, lon_type=lon_type)

        # Snyder 1987, eq. 14-3 to 14-6.
        self.n = (np.sin(lat_1 * DEG2RAD) + np.sin(lat_2 * DEG2RAD)) / 2
        self.C = np.cos(lat_1 * DEG2RAD)**2 + 2 * self.n * np.sin(lat_1 * DEG2RAD)
        self.rho_0 = self._rho(lat_0)

    def _rho(self, lat):
        return np.sqrt(self.C - 2 * self.n * np.sin(lat * DEG2RAD)) / self.n

    def transform(self, lon, lat):
        lon_ = self._standardize(lon)
        # Snyder 1987, eq 14-1 to 14-4
        theta = self.n * lon_
        rho = self._rho(lat)
        return rho*np.sin(theta * DEG2RAD), self.rho_0 - rho*np.cos(theta * DEG2RAD)

    def invert(self, x, y):
        # lon/lat actually x/y
        # Snyder 1987, eq 14-8 to 14-11
        rho = np.sqrt(x**2 + (self.rho_0 - y)**2)
        if self.n >= 0:
            theta = np.arctan2(x, self.rho_0 - y) / DEG2RAD
        else:
            theta = np.arctan2(-x, -(self.rho_0 - y)) / DEG2RAD
        lon = self._unstandardize(theta/self.n)
        lat = np.arcsin((self.C - (rho * self.n)**2)/(2*self.n)) / DEG2RAD
        return lon, lat


class LambertConformal(ConicProjection, Projection):
    """Lambert Conformal conic projection

    LCC is a conic projection with an origin along the lines connecting
    the poles. It preserves angles, but is not equal-area,
    perspective or equistant.

    Its preferred use of for areas with predominant east-west extent
    at higher latitudes.

    As a conic projection, it depends on two standard parallels, i.e.
    intersections of the cone with the sphere. To minimize scale variations,
    these standard parallels should be chosen as small as possible while
    spanning the range in declinations of the data.

    For details, see Snyder (1987, section 15).
    """
    def __init__(self, lon_0, lat_0, lat_1, lat_2, lon_type="ra"):
        """Create Lambert Conformal Conic projection

        Args:
            lon_0: longitude that maps onto x = 0
            lat_0: latitude that maps onto y = 0
            lat_1: lower standard parallel
            lat_2: upper standard parallel (must not be -lat_1)
            lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)
        """
        super(LambertConformal, self).__init__(lon_0, lat_0, lat_1, lat_2, lon_type=lon_type)

        # Snyder 1987, eq. 14-1, 14-2 and 15-1 to 15-3.
        self.dec_max = 89.999
        lat_1 *= DEG2RAD
        lat_2 *= DEG2RAD
        self.n = np.log(np.cos(lat_1)/np.cos(lat_2)) / \
        (np.log(np.tan(np.pi/4 + lat_2/2)/np.tan(np.pi/4 + lat_1/2)))
        self.F = np.cos(lat_1)*(np.tan(np.pi/4 + lat_1/2)**self.n)/self.n
        self.rho_0 = self._rho(lat_0)

    @property
    def poleIsPoint(self):
        # because of dec_max: the pole isn't reached
        self._poleIsPoint = {90: False, -90: False}
        if self.n >= 0:
            self._poleIsPoint[90] = True
        else:
            self._poleIsPoint[-90] = True
        return self._poleIsPoint

    def _rho(self, lat):
        # check that lat is inside of -dec_max .. dec_max
        lat_ = np.array([lat], dtype='f8')
        lat_[lat_ < -self.dec_max] = -self.dec_max
        lat_[lat_ > self.dec_max] = self.dec_max
        return self.F / np.tan(np.pi/4 + lat_[0]/2 * DEG2RAD)**self.n

    def transform(self, lon, lat):
        lon_ = self._standardize(lon)
        theta = self.n * lon_
        rho = self._rho(lat)
        return rho*np.sin(theta * DEG2RAD), self.rho_0 - rho*np.cos(theta * DEG2RAD)

    def invert(self, x, y):
        rho = np.sqrt(x**2 + (self.rho_0 - y)**2) * np.sign(self.n)
        if self.n >= 0:
            theta = np.arctan2(x, self.rho_0 - y) / DEG2RAD
        else:
            theta = np.arctan2(-x, -(self.rho_0 - y)) / DEG2RAD
        lon = self._unstandardize(theta/self.n)
        lat = (2 * np.arctan((self.F/rho)**(1./self.n)) - np.pi/2) / DEG2RAD
        return lon, lat


class Equidistant(ConicProjection, Projection):
    """Equidistant conic projection

    Equistant conic is a projection with an origin along the lines connecting
    the poles. It preserves distances along the map, but is not conformal,
    perspective or equal-area.

    Its preferred use is for smaller areas with predominant east-west extent
    at moderate latitudes.

    As a conic projection, it depends on two standard parallels, i.e.
    intersections of the cone with the sphere.

    For details, see Snyder (1987, section 16).
    """
    def __init__(self, lon_0, lat_0, lat_1, lat_2, lon_type="ra"):
        """Create Equidistant Conic projection

        Args:
            lon_0: longitude that maps onto x = 0
            lat_0: latitude that maps onto y = 0
            lat_1: lower standard parallel
            lat_2: upper standard parallel (must not be +-lat_1)
            lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)
        """
        super(Equidistant, self).__init__(lon_0, lat_0, lat_1, lat_2, lon_type=lon_type)

        # Snyder 1987, eq. 14-3 to 14-6.
        self.n = (np.cos(lat_1 * DEG2RAD) - np.cos(lat_2 * DEG2RAD)) / (lat_2  - lat_1) / DEG2RAD
        self.G = np.cos(lat_1 * DEG2RAD)/self.n + (lat_1 * DEG2RAD)
        self.rho_0 = self._rho(lat_0)

    def _rho(self, lat):
        return self.G - (lat * DEG2RAD)

    def transform(self, lon, lat):
        lon_ = self._standardize(lon)
        # Snyder 1987, eq 16-1 to 16-4
        theta = self.n * lon_
        rho = self._rho(lat)
        return rho*np.sin(theta * DEG2RAD), self.rho_0 - rho*np.cos(theta * DEG2RAD)

    def invert(self, x, y):
        # Snyder 1987, eq 14-10 to 14-11
        rho = np.sqrt(x**2 + (self.rho_0 - y)**2) * np.sign(self.n)
        if self.n >= 0:
            theta = np.arctan2(x, self.rho_0 - y) / DEG2RAD
        else:
            theta = np.arctan2(-x, -(self.rho_0 - y)) / DEG2RAD
        lon = self._unstandardize(theta/self.n)
        lat = (self.G - rho)/ DEG2RAD
        return lon, lat


class Hammer(Projection):
    """Hammer projection

    Hammer's 2:1 ellipse modification of the Lambert azimuthal equal-area
    projection.

    Its preferred use is for all-sky maps with an emphasis on low latitudes.
    It reduces the distortion at the outer meridians and has an elliptical
    outline. The only free parameter is the reference RA `lon_0`.

    For details, see Snyder (1987, section 24).
    """
    def __init__(self, lon_0=0, lon_type="ra"):
        """Create Hammer projection

        Args:
            lon_0: longitude that maps onto x = 0
            lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)
        """
        super(Hammer, self).__init__(lon_0, lon_type)

    def transform(self, lon, lat):
        lon_ = self._standardize(lon)
        x = 2*np.sqrt(2)*np.cos(lat * DEG2RAD) * np.sin(lon_/2 * DEG2RAD)
        y = np.sqrt(2)*np.sin(lat * DEG2RAD)
        denom = np.sqrt(1+ np.cos(lat * DEG2RAD) * np.cos(lon_/2 * DEG2RAD))
        return x/denom, y/denom

    def invert(self, x, y):
        dz = x*x/16 + y*y/4
        z = np.sqrt(1- dz)
        lat = np.arcsin(z*y) / DEG2RAD
        lon = 2*np.arctan(z*x / (2*(2*z*z - 1))) / DEG2RAD
        lon = self._unstandardize(lon)
        return lon, lat


class Mollweide(Projection):
    """Mollweide projection

    Mollweide elliptical equal-area projection. It is used for all-sky maps,
    but it introduces strong distortions at the outer meridians.
    The only free parameter is the reference RA `lon_0`.

    For details, see Snyder (1987, section 31).
    """
    def __init__(self, lon_0=0, lon_type="ra"):
        """Create Mollweide projection

        Args:
            lon_0: longitude that maps onto x = 0
            lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)
        """
        super(Mollweide, self).__init__(lon_0, lon_type)
        self.sqrt2 = np.sqrt(2)

    def transform(self, lon, lat):
        # Snyder p. 251
        lon_ = self._standardize(lon)
        theta_ = self.theta(lat)
        x = 2*self.sqrt2 / np.pi * (lon_ * DEG2RAD) * np.cos(theta_)
        y = self.sqrt2 * np.sin(theta_)
        return x, y

    def theta(self, lat, eps=1e-6, maxiter=100):
        # Snyder 1987 p. 251
        # Newon scheme to solve for theta given phi (=Dec)
        lat_ = lat * DEG2RAD
        t0 = lat_
        mask = np.abs(lat_) < np.pi/2
        if mask.any():
            t = t0[mask]
            for it in range(maxiter):
                f = 2*t + np.sin(2*t) - np.pi*np.sin(lat_[mask])
                fprime = 2 + 2*np.cos(2*t)
                t_ = t - f / fprime
                if (np.abs(t - t_) < eps).all():
                    t = t_
                    break
                t = t_
            t0[mask] = t
        return t0

    def invert(self, x, y):
        theta_ = np.arcsin(y/self.sqrt2)
        lon = self._unstandardize(np.pi*x/(2*self.sqrt2*np.cos(theta_)) / DEG2RAD)
        lat = np.arcsin((2*theta_ + np.sin(2*theta_))/np.pi) / DEG2RAD
        return lon, lat


class EckertIV(Projection):
    """Eckert IV projection

    Eckert's IV equal-area projection is used for all-sky maps.
    The only free parameter is the reference RA `lon_0`.

    For details, see Snyder (1987, section 32).
    """
    def __init__(self, lon_0=0, lon_type="ra"):
        """Create Eckert IV projection

        Args:
            lon_0: longitude that maps onto x = 0
            lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)
        """
        super(EckertIV, self).__init__(lon_0, lon_type)
        self.c1 = 2 / np.sqrt(4*np.pi + np.pi**2)
        self.c2 = 2 * np.sqrt(1/(4/np.pi + 1))

    def transform(self, lon, lat):
        lon_ = self._standardize(lon)
        t = self.theta(lat)
        x = self.c1 * lon_ *DEG2RAD * (1 + np.cos(t))
        y = self.c2 * np.sin(t)
        return x, y

    def invert(self, x, y):
        t = np.arcsin(y / self.c2)
        lon = self._unstandardize(x / (1+np.cos(t)) / self.c1 / DEG2RAD)
        lat = np.arcsin(y / self.c2) / DEG2RAD
        return lon, lat

    def theta(self, lat, eps=1e-6, maxiter=100):
        # Snyder 1993 p. 195
        # Newon scheme to solve for theta given phi (=Dec)
        lat_ = lat * DEG2RAD
        t = lat_
        for it in range(maxiter):
            f = t + np.sin(t)*np.cos(t) + 2*np.sin(t) - (2+np.pi/2)*np.sin(lat_)
            fprime = 1 + np.cos(t)**2 - np.sin(t)**2 + 2*np.cos(t)
            t_ = t - f / fprime
            if (np.abs(t - t_) < eps).all():
                t = t_
                break
            t = t_
        return t


class WagnerI(Projection):
    """Wagner I projection

    Wagners's I equal-area projection is used for all-sky maps.
    The only free parameter is the reference RA `lon_0`.

    For details, see Snyder (1993, p. 204).
    """
    def __init__(self, lon_0=0, lon_type="ra"):
        """Create WagnerI projection

        Args:
            lon_0: longitude that maps onto x = 0
            lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)
        """
        super(WagnerI, self).__init__(lon_0, lon_type)
        self.c1 = 2 / 3**0.75
        self.c2 = 3**0.25
        self.c3 = np.sqrt(3)/2

    def transform(self, lon, lat):
        lon_ = self._standardize(lon)
        t = np.arcsin(self.c3*np.sin(lat * DEG2RAD))
        x = self.c1 * lon_ *DEG2RAD * np.cos(t)
        y = self.c2 * t
        return x, y

    def invert(self, x, y):
        t = y / self.c2
        lon = self._unstandardize(x / np.cos(t) / self.c1 / DEG2RAD)
        lat = np.arcsin(np.sin(t) / self.c3) / DEG2RAD
        return lon, lat


class WagnerIV(Projection):
    """Wagner IV projection

    Wagner's IV equal-area projection is used for all-sky maps.
    The only free parameter is the reference RA `lon_0`.

    For details, see Snyder (1993, p. 204).
    """
    def __init__(self, lon_0=0, lon_type="ra"):
        """Create WagnerIV projection

        Args:
            lon_0: longitude that maps onto x = 0
            lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)
        """
        super(WagnerIV, self).__init__(lon_0, lon_type)
        self.c1 = 0.86310
        self.c2 = 1.56548
        self.c3 = (4*np.pi + 3*np.sqrt(3)) / 6

    def transform(self, lon, lat):
        lon_ = self._standardize(lon)
        t = self.theta(lat)
        x = self.c1 * lon_ * DEG2RAD * np.cos(t)
        y = self.c2 * np.sin(t)
        return x, y

    def invert(self, x, y):
        t = np.arcsin(y / self.c2)
        lon = self._unstandardize(x / np.cos(t) / self.c1 / DEG2RAD)
        lat = np.arcsin(y / self.c2) / DEG2RAD
        return lon, lat

    def theta(self, lat, eps=1e-6, maxiter=100):
        # Newon scheme to solve for theta given phi (=Dec)
        lat_ = lat * DEG2RAD
        t0 = np.zeros(lat_.shape)
        mask = np.abs(lat_) < np.pi/2
        if mask.any():
            t = t0[mask]
            for it in range(maxiter):
                f = 2*t + np.sin(2*t) - self.c3*np.sin(lat_[mask])
                fprime = 2 + 2*np.cos(2*t)
                t_ = t - f / fprime
                if (np.abs(t - t_) < eps).all():
                    t = t_
                    break
                t = t_
            t0[mask] = t
        t0[~mask] = np.sign(lat[~mask]) * np.pi/3 # maximum value
        return t0


class WagnerVII(Projection):
    """Wagner VII projection

    WagnerVII equal-area projection is used for all-sky maps.
    The only free parameter is the reference RA `lon_0`.

    For details, see Snyder (1993, p. 237).
    """
    def __init__(self, lon_0=0, lon_type="ra"):
        """Create WagnerVII projection

        Args:
            lon_0: longitude that maps onto x = 0
            lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)
        """
        super(WagnerVII, self).__init__(lon_0, lon_type)
        self.c1 = 2.66723
        self.c2 = 1.24104
        self.c3 = np.sin(65 * DEG2RAD)

    def transform(self, lon, lat):
        lon_ = self._standardize(lon)
        theta = np.arcsin(self.c3 * np.sin(lat * DEG2RAD))
        alpha = np.arccos(np.cos(theta)*np.cos(lon_ * DEG2RAD/3))
        x = self.c1 * np.cos(theta) * np.sin(lon_ * DEG2RAD / 3) / np.cos(alpha/2)
        y = self.c2 * np.sin(theta) / np.cos(alpha/2)
        return x, y


class McBrydeThomasFPQ(Projection):
    """McBryde-Thomas Flat-Polar Quartic projection

    McBrydeThomasFPQ equal-area projection is used for all-sky maps.
    The only free parameter is the reference RA `lon_0`.

    For details, see Snyder (1993, p. 211).
    """
    def __init__(self, lon_0=0, lon_type="ra"):
        """Create McBryde-Thomas Flat-Polar Quartic projection

        Args:
            lon_0: longitude that maps onto x = 0
            lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)
        """
        super(McBrydeThomasFPQ, self).__init__(lon_0, lon_type)
        self.c1 = 1 / np.sqrt(3*np.sqrt(2) + 6)
        self.c2 = 2 * np.sqrt(3 / (2 + np.sqrt(2)))
        self.c3 = 1 + np.sqrt(2) / 2

    def transform(self, lon, lat):
        lon_ = self._standardize(lon)
        t = self.theta(lat)
        x = self.c1 * lon_ * DEG2RAD * (1 + 2*np.cos(t)/np.cos(t/2))
        y = self.c2 * np.sin(t/2)
        return x, y

    def invert(self, x, y):
        t = 2*np.arcsin(y / self.c2)
        lon = self._unstandardize(x / (1 + 2*np.cos(t)/np.cos(t/2)) / self.c1 / DEG2RAD)
        lat = np.arcsin((np.sin(t/2) + np.sin(t))/ self.c3) / DEG2RAD
        return lon, lat

    def theta(self, lat, eps=1e-6, maxiter=100):
        # Newon scheme to solve for theta given phi (=Dec)
        lat_ = lat * DEG2RAD
        t = lat_
        for it in range(maxiter):
            f = np.sin(t/2) + np.sin(t) - self.c3*np.sin(lat_)
            fprime = np.cos(t/2)/2 + np.cos(t)
            t_ = t - f / fprime
            if (np.abs(t - t_) < eps).all():
                t = t_
                break
            t = t_
        return t


class HyperElliptical(Projection):
    """Hyperelliptical projection

    The outline of the map follows the equation
        |x/a|^k + |y/b|^k = gamma^k
    The parameter alpha is a weight between cylindrical equal-area (alpha=0)
    and sinosoidal projections.

    The projection does not have a closed form for either forward or backward
    transformation and this therefore computationally expensive.

    See Snyder (1993, p. 220) for details.
    """
    def __init__(self, lon_0, alpha, k, gamma, lon_type="ra"):
        """Create Hyperelliptical projection

        Args:
            lon_0: longitude that maps onto x = 0
            alpha: cylindrical-sinosoidal weight
            k: hyperelliptical exponent
            gamma: hyperelliptical scale
            lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)
        """
        super(HyperElliptical, self).__init__(lon_0, lon_type)
        self.alpha = alpha
        self.k = k
        self.gamma = gamma
        self.gamma_pow_k = np.abs(gamma)**k
        self.affine = np.sqrt(2 * self.gamma / np.pi)

    def transform(self, lon, lat):
        lon_ = self._standardize(lon)
        y = self.Y(np.sin(np.abs(lat * DEG2RAD)))
        x = lon_ * DEG2RAD * (self.alpha + (1 - self.alpha) / self.gamma * self.elliptic(y)) * self.affine
        y *= np.sign(lat) / self.affine
        return x, y

    def invert(self, x, y):
        y_ = y * self.affine
        sinphi = self.sinPhiDiff(y_, 0)
        lat = np.sign(y) * np.arcsin(sinphi) / DEG2RAD

        lon = x / self.affine / (self.alpha + (1 - self.alpha) / self.gamma * self.elliptic(y_)) / DEG2RAD
        lon = self._unstandardize(lon)
        return  lon, lat

    def elliptic(self, y):
        """Returns (gamma^k - y^k)^1/k
        """
        y_,isArray = _toArray(y)

        f = (self.gamma_pow_k - y_**self.k)**(1/self.k)
        f[y_ < 0 ] = self.gamma

        if isArray:
            return f
        else:
            return f[0]

    def elliptic_scalar(self, y):
        """Returns (gamma^k - y^k)^1/k
        """
        # needs to be fast for integrator, hence non-vectorized version
        if y < 0:
            return self.gamma
        return (self.gamma_pow_k - y**self.k)**(1/self.k)

    def z(self, y):
        """Returns int_0^y (gamma^k - y_^k)^1/k dy_
        """
        if hasattr(y, "__iter__"):
            return np.array([self.z(_) for _ in y])

        f = scipy.integrate.quad(self.elliptic_scalar, 0, y)[0]

        # check integration errors ofat the limits
        lim1 = self.gamma * (self.alpha*y - 1) / (self.alpha - 1)
        lim2 = self.gamma * self.alpha*y / (self.alpha - 1)
        if f < lim2:
            return lim2
        if f > lim1:
            return lim1
        return f

    def sinPhiDiff(self, y, sinphi):
        return self.alpha*y - (self.alpha - 1) / self.gamma * self.z(y) - sinphi

    def Y(self, sinphi, eps=1e-5, max_iter=30):
        if hasattr(sinphi, "__iter__"):
            return np.array([self.Y(_) for _ in sinphi])

        y, it, delta = 0.01, 0, 2*eps
        while it < max_iter and np.abs(delta) > eps:
            delta = self.sinPhiDiff(y, sinphi) / (self.alpha + (1 - self.alpha) / self.gamma * self.elliptic(y))
            y -= delta

            if y >= self.gamma:
                return self.gamma
            if y <= 0:
                return 0.
            it += 1
        return y

class Tobler(HyperElliptical):
    """Tobler hyperelliptical projection

    Tobler's cylindrical equal-area projection is a specialization of
    `HyperElliptical` with parameters `alpha=0`, `k=2.5`, `gamma=1.183136`.

    See Snyder (1993, p. 220) for details.
    """
    def __init__(self, lon_0=0, lon_type="ra"):
        """Create Tobler projection

        Args:
            lon_0: longitude that maps onto x = 0
            lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)
        """
        alpha, k, gamma = 0., 2.5, 1.183136
        super(Tobler, self).__init__(lon_0, alpha, k, gamma, lon_type=lon_type)


class EqualEarth(Projection):
    """Equal Earth projection

    The Equal Earth projection is a pseudo-cylindrical equal-area projection
    with modest distortion.

    See https://doi.org/10.1080/13658816.2018.1504949 for details.
    """
    def __init__(self, lon_0=0, lon_type="ra"):
        """Create Equal Earth projection

        Args:
            lon_0: longitude that maps onto x = 0
            lon_type: type of longitude, "lon" or "ra" (see `BaseProjection`)
        """
        super(EqualEarth, self).__init__(lon_0, lon_type)
        self.A1 = 1.340264
        self.A2 = -0.081106
        self.A3 = 0.000893
        self.A4 = 0.003796
        self.sqrt3 = np.sqrt(3)

    def transform(self, lon, lat):
        lon_ = self._standardize(lon)

        t = np.arcsin(self.sqrt3/2 * np.sin(lat * DEG2RAD))
        t2 = t*t
        t6 = t2*t2*t2
        x = 2/3*self.sqrt3 * lon_ * DEG2RAD * np.cos(t) / (self.A1 + 3*self.A2*t2 + t6*(7*self.A3 + 9*self.A4*t2))
        y = t*(self.A1 + self.A2*t2 + t6*(self.A3 + self.A4*t2))
        return x, y
