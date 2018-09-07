import numpy as np
DEG2RAD = np.pi/180

class Projection(object):
    """Projection base class

    Every projection needs to implement three methods:
    * `transform(self, ra, dec)`: mapping from ra/dec to map x/y
    * `invert(self, x, y)`: the inverse mapping from x/y to ra/dec
    * `contains(self, x, y)`: whether x/y is inside of map region

    All three methods accept either single number or arrays and return accordingly.
    """
    def transform(self, ra, dec):
        """Convert RA/Dec into map coordinates

        Args:
            ra:  float or array of floats
            dec: float or array of floats

        Returns:
            x,y with the same format as ra/dec
        """
        pass

    def invert(self, x, y):
        """Convert map coordinates into RA/Dec

        Args:
            x:  float or array of floats
            y: float or array of floats

        Returns:
            RA,Dec with the same format as x/y
        """
        pass

    def contains(self, x, y):
        """Test if x/y is a valid set of map coordinates

        Args:
            x:  float or array of floats
            y: float or array of floats

        Returns:
            Bool array with the same format as x/y
        """
        pass

    def _wrapRA(self, ra):
        ra_ = np.array([ra - self.ra_0]) * -1 # inverse for RA
        # check that ra_ is between -180 and 180 deg
        ra_[ra_ < -180 ] += 360
        ra_[ra_ > 180 ] -= 360
        return ra_[0]


class AlbersEqualAreaConic(Projection):
    def __init__(self, ra_0, dec_0, dec_1, dec_2):
        """Albers Equal-Area projection

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

        Args:
            ra_0: RA that maps onto x = 0
            dec_0: Dec that maps onto y = 0
            dec_1: lower standard parallel
            dec_2: upper standard parallel (must not be -dec_1)
        """
        self.ra_0 = ra_0
        self.dec_0 = dec_0
        self.dec_1 = dec_1
        self.dec_2 = dec_2

        # Snyder 1987, eq. 14-3 to 14-6.
        self.n = (np.sin(dec_1 * DEG2RAD) + np.sin(dec_2 * DEG2RAD)) / 2
        self.C = np.cos(dec_1 * DEG2RAD)**2 + 2 * self.n * np.sin(dec_1 * DEG2RAD)
        self.rho_0 = self._rho(dec_0)

    def _rho(self, dec):
        return np.sqrt(self.C - 2 * self.n * np.sin(dec * DEG2RAD)) / self.n

    def transform(self, ra, dec):
        ra_ = self._wrapRA(ra)
        # Snyder 1987, eq 14-1 to 14-4
        theta = self.n * ra_
        rho = self._rho(dec)
        return rho*np.sin(theta * DEG2RAD), self.rho_0 - rho*np.cos(theta * DEG2RAD)

    def contains(self, x, y):
        rho = np.sqrt(x**2 + (self.rho_0 - y)**2)
        inside = np.abs((self.C - (rho * self.n)**2)/(2*self.n)) <= 1
        if self.n >= 0:
            theta = np.arctan2(x, self.rho_0 - y) / DEG2RAD
        else:
            theta = np.arctan2(-x, -(self.rho_0 - y)) / DEG2RAD
        wedge = np.abs(theta) < np.abs(self.n*180)
        return inside & wedge

    def invert(self, x, y):
        # ra/dec actually x/y
        # Snyder 1987, eq 14-8 to 14-11
        rho = np.sqrt(x**2 + (self.rho_0 - y)**2)
        if self.n >= 0:
            theta = np.arctan2(x, self.rho_0 - y) / DEG2RAD
        else:
            theta = np.arctan2(-x, -(self.rho_0 - y)) / DEG2RAD
        return self.ra_0 - theta/self.n, np.arcsin((self.C - (rho * self.n)**2)/(2*self.n)) / DEG2RAD

    def __repr__(self):
        return "AlbersEqualArea(%r, %r, %r, %r)" % (self.ra_0, self.dec_0, self.dec_1, self.dec_2)

class LambertConformalConic(Projection):
    def __init__(self, ra_0, dec_0, dec_1, dec_2):
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

        Args:
            ra_0: RA that maps onto x = 0
            dec_0: Dec that maps onto y = 0
            dec_1: lower standard parallel
            dec_2: upper standard parallel (must not be -dec_1)
        """
        self.ra_0 = ra_0
        self.dec_0 = dec_0
        self.dec_1 = dec_1
        self.dec_2 = dec_2

        # Snyder 1987, eq. 14-1, 14-2 and 15-1 to 15-3.
        self.dec_max = 89.99

        dec_1 *= DEG2RAD
        dec_2 *= DEG2RAD
        self.n = np.log(np.cos(dec_1)/np.cos(dec_2)) / \
        (np.log(np.tan(np.pi/4 + dec_2/2)/np.tan(np.pi/4 + dec_1/2)))
        self.F = np.cos(dec_1)*(np.tan(np.pi/4 + dec_1/2)**self.n)/self.n
        self.rho_0 = self._rho(dec_0)

    def _rho(self, dec):
        # check that dec is inside of -dec_max .. dec_max
        dec_ = np.array([dec], dtype='f8')
        dec_[dec_ < -self.dec_max] = -self.dec_max
        dec_[dec_ > self.dec_max] = self.dec_max
        return self.F / np.tan(np.pi/4 + dec_[0]/2 * DEG2RAD)**self.n

    def transform(self, ra, dec):
        ra_ = self._wrapRA(ra)
        theta = self.n * ra_
        rho = self._rho(dec)
        return rho*np.sin(theta * DEG2RAD), self.rho_0 - rho*np.cos(theta * DEG2RAD)

    def contains(self, x, y):
        rho = np.sqrt(x**2 + (self.rho_0 - y)**2) * np.sign(self.n)
        inside = np.abs(rho) < max(np.abs(self._rho(self.dec_max)), np.abs(self._rho(-self.dec_max)))
        if self.n >= 0:
            theta = np.arctan2(x, self.rho_0 - y) / DEG2RAD
        else:
            theta = np.arctan2(-x, -(self.rho_0 - y)) / DEG2RAD
        wedge = np.abs(theta) < np.abs(self.n*180)
        return inside & wedge

    def invert(self, x, y):
        rho = np.sqrt(x**2 + (self.rho_0 - y)**2) * np.sign(self.n)
        if self.n >= 0:
            theta = np.arctan2(x, self.rho_0 - y) / DEG2RAD
        else:
            theta = np.arctan2(-x, -(self.rho_0 - y)) / DEG2RAD
        return self.ra_0 - theta/self.n, (2 * np.arctan((self.F/rho)**(1./self.n)) - np.pi/2) / DEG2RAD

    def __repr__(self):
        return "LambertConformal(%r, %r, %r, %r)" % (self.ra_0, self.dec_0, self.dec_1, self.dec_2)


class EquidistantConic(Projection):
    def __init__(self, ra_0, dec_0, dec_1, dec_2):
        """Equidistant conic projection

        Equistant conic is a projection with an origin along the lines connecting
        the poles. It preserves distances along the map, but is not conformal,
        perspective or equal-area.

        Its preferred use is for smaller areas with predominant east-west extent
        at moderate latitudes.

        As a conic projection, it depends on two standard parallels, i.e.
        intersections of the cone with the sphere.

        For details, see Snyder (1987, section 16).

        Args:
            ra_0: RA that maps onto x = 0
            dec_0: Dec that maps onto y = 0
            dec_1: lower standard parallel
            dec_2: upper standard parallel (must not be +-dec_1)
        """
        self.ra_0 = ra_0
        self.dec_0 = dec_0
        self.dec_1 = dec_1
        self.dec_2 = dec_2

        # Snyder 1987, eq. 14-3 to 14-6.
        self.n = (np.cos(dec_1 * DEG2RAD) - np.cos(dec_2 * DEG2RAD)) / (dec_2  - dec_1) / DEG2RAD
        self.G = np.cos(dec_1 * DEG2RAD)/self.n + (dec_1 * DEG2RAD)
        self.rho_0 = self._rho(dec_0)

    def _rho(self, dec):
        return self.G - (dec * DEG2RAD)

    def transform(self, ra, dec):
        ra_ = self._wrapRA(ra)
        # Snyder 1987, eq 16-1 to 16-4
        theta = self.n * ra_
        rho = self._rho(dec)
        return rho*np.sin(theta * DEG2RAD), self.rho_0 - rho*np.cos(theta * DEG2RAD)

    def contains(self, x, y):
        rho = np.sqrt(x**2 + (self.rho_0 - y)**2) * np.sign(self.n)
        rho_min = np.abs(self._rho(90))
        rho_max = np.abs(self._rho(-90))
        if rho_min > rho_max:
            rho_min, rho_max = rho_max, rho_min
        inside = (np.abs(rho) < rho_max) & (np.abs(rho) > rho_min)
        if self.n >= 0:
            theta = np.arctan2(x, self.rho_0 - y) / DEG2RAD
        else:
            theta = np.arctan2(-x, -(self.rho_0 - y)) / DEG2RAD
        wedge = np.abs(theta) < np.abs(self.n*180)
        return inside & wedge

    def invert(self, x, y):
        # Snyder 1987, eq 14-10 to 14-11
        rho = np.sqrt(x**2 + (self.rho_0 - y)**2) * np.sign(self.n)
        if self.n >= 0:
            theta = np.arctan2(x, self.rho_0 - y) / DEG2RAD
        else:
            theta = np.arctan2(-x, -(self.rho_0 - y)) / DEG2RAD
        return self.ra_0 - theta/self.n, (self.G - rho)/ DEG2RAD

    def __repr__(self):
        return "Equidistant(%r, %r, %r, %r)" % (self.ra_0, self.dec_0, self.dec_1, self.dec_2)


class Hammer(Projection):
    def __init__(self, ra_0):
        """Hammer projection

        Hammer's 2:1 ellipse modification of The Lambert azimuthal equal-area
        projection.

        Its preferred use is for all-sky maps with an emphasis on low latitudes.
        It reduces the distortion at the outer meridians and has an elliptical
        outline. The only free parameter is the reference RA `ra_0`.

        For details, see Snyder (1987, section 24).
        """
        self.ra_0 = ra_0

    def transform(self, ra, dec):
        ra_ = self._wrapRA(ra)
        x = 2*np.sqrt(2)*np.cos(dec * DEG2RAD) * np.sin(ra_/2 * DEG2RAD)
        y = np.sqrt(2)*np.sin(dec * DEG2RAD)
        denom = np.sqrt(1+ np.cos(dec * DEG2RAD) * np.cos(ra_/2 * DEG2RAD))
        return x/denom, y/denom

    def invert(self, x, y):
        dz = x*x/16 + y*y/4
        z = np.sqrt(1- dz)
        phi = np.arcsin(z*y) / DEG2RAD
        lmbda = 2*np.arctan(z*x / (2*(2*z*z - 1))) / DEG2RAD
        return self.ra_0 - lmbda, phi

    def contains(self, x, y):
        dz = x*x/16 + y*y/4
        return dz <= 0.5

    def __repr__(self):
        return "Hammer(%r)" % self.ra_0

class HyperElliptic(Projection):
    def __init__(self, ra_0, alpha, k, gamma):
        self.ra_0 = ra_0
        self.alpha = alpha
        self.k = k
        self.gamma = gamma

        self.G = 1 / self.z(1)
        self.n = 1000
        m = (1 + 1e-8) * self.G
        self.approx = [ self.z(i/self.n) * m for i in range(self.n+1) ]
        self.ratio = 2 * self.Y(1) / np.pi * self.G / self.gamma;


    def transform(self, ra, dec):
        ra_ = self._wrapRA(ra)
        y = self.Y(np.abs(np.sin(dec * DEG2RAD)))
        x = self.elliptic(y) * ra_ * DEG2RAD
        y *= np.sign(dec)/self.ratio
        if hasattr(x, "__iter__") and not hasattr(y, "__iter__"):
            y = y*np.ones(len(x))

        return x, y

    def invert(self, x, y):
        y_ = y*self.ratio
        dec = np.sign(y_) * np.arcsin(self.z(np.abs(y_)) * self.G) / DEG2RAD
        return self.ra_0 - x / self.elliptic(np.abs(y_)) / DEG2RAD, dec

    def contains(self, x, y):
        return np.abs(self.ratio * y) < 1

    def elliptic(self, f):
        return self.alpha + (1 - self.alpha) * (1 - f**self.k)**(1/self.k)

    def z(self, f):
        if hasattr(f, "__iter__"):
            return np.array([self.z(f[i]) for i in range(len(f))])

        import scipy.integrate as integrate
        result = integrate.quad(self.elliptic, 0, f)
        return result[0]

    def Y(self, sinphi):
        if hasattr(sinphi, "__iter__"):
            return np.array([self.Y(sinphi[i]) for i in range(len(sinphi))])

        rmin, rmax, r = 0, self.n, self.n >> 1
        while True:
            if self.approx[r] > sinphi:
                rmax = r
            else:
                rmin = r
            r = (rmin + rmax) >> 1
            if r <= rmin:
                break

        u = self.approx[r + 1] - self.approx[r]
        if u:
            u = (sinphi - self.approx[r + 1]) / u
        return (r + 1 + u) / self.n

class Tobler(HyperElliptic):
    def __init__(self, ra_0):
        alpha, k, gamma = 0, 2.5, 1.183136
        super(Tobler, self).__init__(ra_0, alpha, k, gamma)

class Mollweide(HyperElliptic):
    def __init__(self, ra_0):
        alpha, k, gamma = 0, 2, 1.2731
        super(Mollweide, self).__init__(ra_0, alpha, k, gamma)
