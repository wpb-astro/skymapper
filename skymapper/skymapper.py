import matplotlib.pyplot as plt
import numpy as np

# python 3 compatible
try:
    xrange
except NameError:
    xrange = range

class ConicProjection(object):
    def __init__(self, ra_0, dec_0, dec_1, dec_2):
        """Base class for conic projections.

        A conic projection depends on two standard parallels, i.e.
        intersections of the cone with the sphere.
        For details, see Snyder (1987, page 97ff).

        Args:
            ra_0: RA that maps onto x = 0
            dec_0: Dec that maps onto y = 0
            dec_1: lower standard parallel
            dec_2: upper standard parallel (must not be -dec_1)
        """
        self.ra_0 = ra_0
        self.dec_0 = dec_0
        self.dec_1 = dec_1 # dec1 and dec2 only needed for __repr__
        self.dec_2 = dec_2
        self.deg2rad = np.pi/180

    def _wrapRA(self, ra):
        ra_ = np.array([ra - self.ra_0]) * -1 # inverse for RA
        # check that ra_ is between -180 and 180 deg
        ra_[ra_ < -180 ] += 360
        ra_[ra_ > 180 ] -= 360
        return ra_[0]

    def getMeridianPatches(self, meridians, **kwargs):
        """Get meridian lines in matplotlib format.

        Meridian lines in conics are circular arcs, appropriate
        matplotlib.patches will be return

        Args:
            meridians: list of declinations
            **kwargs: matplotlib.patches.Arc parameters

        Returns:
            matplotlib.PatchCollection
        """
        from matplotlib.patches import Arc
        from matplotlib.collections import PatchCollection

        # get opening angle
        origin = (0, self.rho_0)
        top_left = self.__call__(self.ra_0 - 180, 90)
        angle_limit = 180 - np.arctan2(origin[0]-top_left[0], origin[1]-top_left[1])/self.deg2rad
        angle = 90
        if self.n < 0:
            angle = -90
            angle_limit = 180 - angle_limit

        # get radii
        _, y = self.__call__(self.ra_0, meridians)
        radius = np.abs(self.rho_0 - y)

        patches = [Arc(origin, 2*radius[m], 2*radius[m], angle=angle, theta1=-angle_limit, theta2=angle_limit, **kwargs) for m in xrange(len(meridians))]
        return PatchCollection(patches, match_original=True, zorder=patches[0].zorder)

    def getParallelPatches(self, parallels, **kwargs):
        """Get parallel lines in matplotlib format.

        Parallel lines in conics are straight, appropriate
        matplotlib.patches will be returned.

        Args:
            meridians: list of rectascensions
            **kwargs: matplotlib.collection.LineCollection parameters

        Returns:
            matplotlib.LineCollection
        """

        # remove duplicates
        parallels_ = np.unique(parallels % 360)

        # the outer boundaries need to be duplicated because the same
        # parallel appear on the left and the right side of the map
        if self.ra_0 < 180:
            outer = self.ra_0 - 180
        else:
            outer = self.ra_0 + 180
        parallels_ = np.array(list(parallels_) + [outer])

        from matplotlib.collections import LineCollection
        top = self.__call__(parallels_, 90)
        bottom = self.__call__(parallels_, -90)
        x_ = np.dstack((top[0], bottom[0]))[0]
        y_ = np.dstack((top[1], bottom[1]))[0]
        return LineCollection(np.dstack((x_, y_)), color='k', **kwargs)

    def findIntersectionAtX(self, x, ylim, ra=None, dec=None):
        """Find intersection of meridian or parallel with a vertical line.

        Args:
            x: x coordinate of the vertical line
            ylim: range in y for the vertical line
            ra: if not None, search for a parallel at this ra to intersect
            dec: if not None, search for a meridian at this dec to intersect

        Returns:
            float or None (if not solution was found)
        """
        if dec is not None:
            # analytic solution for intersection of circle with line at x
            r = np.abs(self.rho_0 - self.__call__(self.ra_0, dec)[1])
            if np.abs(x) > np.abs(r):
                return None
            if self.rho_0 >= 0:
                return self.rho_0 - np.sqrt(r**2 - x**2)
            else:
                return np.sqrt(r**2 - x**2) + self.rho_0
        if ra is not None:
            # analytic solution for intersection of line with line at y
            top = self.__call__(ra, 90)
            bottom = self.__call__(ra, -90)
            delta = (top[0] - bottom[0], top[1] - bottom[1])
            y = bottom[1] + (x - bottom[0]) * delta[1] / delta[0]
            if y >= ylim[0] and y <= ylim[1]:
                return y
            else:
                return None
        raise NotImplementedError("specify either RA or Dec")

    def findIntersectionAtY(self, y, xlim, ra=None, dec=None):
        """Find intersection of meridian or parallel with a horizontal line.

        Args:
            y: y coordinate of the horizontal line
            xlim: range in x for the horizontal line
            ra: if not None, search for a parallel at this ra to intersect
            dec: if not None, search for a meridian at this dec to intersect

        Returns:
            float or None (if not solution was found)
        """
        if dec is not None:
            # analytic solution for intersection of circle with line at y
            r = np.abs(self.rho_0 - self.__call__(self.ra_0, dec)[1])
            if np.abs(y) > np.abs(r):
                return None
            if self.rho_0 >= 0:
                return self.rho_0 - np.sqrt(r**2 - y**2)
            else:
                return np.sqrt(r**2 - y**2) + self.rho_0
        if ra is not None:
            # analytic solution for intersection of line with line at y
            top = self.__call__(ra, 90)
            bottom = self.__call__(ra, -90)
            if y > top[1] or y < bottom[1]:
                return None
            delta = (top[0] - bottom[0], top[1] - bottom[1])
            x = bottom[0] + (y - bottom[1]) * delta[0] / delta[1]
            if x >= xlim[0] and x <= xlim[1]:
                return x
            else:
                return None
        raise NotImplementedError("specify either RA or Dec")


class AlbersEqualAreaProjection(ConicProjection):
    def __init__(self, ra_0, dec_0, dec_1, dec_2):
        """Albers Equal-Area projection.

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
        ConicProjection.__init__(self, ra_0, dec_0, dec_1, dec_2)

        # Snyder 1987, eq. 14-3 to 14-6.
        self.n = (np.sin(dec_1 * self.deg2rad) + np.sin(dec_2 * self.deg2rad)) / 2
        self.C = np.cos(dec_1 * self.deg2rad)**2 + 2 * self.n * np.sin(dec_1 * self.deg2rad)
        self.rho_0 = self._rho(dec_0)

    def _rho(self, dec):
        return np.sqrt(self.C - 2 * self.n * np.sin(dec * self.deg2rad)) / self.n

    def __call__(self, ra, dec, inverse=False):
        """Convert RA/Dec into map coordinates, or the reverse.

        Args:
            ra:  float or array of floats
            dec: float or array of floats
            inverse: if True, convert from map coordinates to RA/Dec

        Returns:
            x,y with the same format as ra/dec
        """
        if not inverse:
            ra_ = self._wrapRA(ra)
            # Snyder 1987, eq 14-1 to 14-4
            theta = self.n * ra_
            rho = self._rho(dec)
            return rho*np.sin(theta * self.deg2rad), self.rho_0 - rho*np.cos(theta * self.deg2rad)
        else:
            # ra/dec actually x/y
            # Snyder 1987, eq 14-8 to 14-11
            rho = np.sqrt(ra**2 + (self.rho_0 - dec)**2)
            if self.n >= 0:
                theta = np.arctan2(ra, self.rho_0 - dec) / self.deg2rad
            else:
                theta = np.arctan2(-ra, -(self.rho_0 - dec)) / self.deg2rad
            return self.ra_0 - theta/self.n, np.arcsin((self.C - (rho * self.n)**2)/(2*self.n)) / self.deg2rad

    def __repr__(self):
        return "AlbersEqualAreaProjection(%r, %r, %r, %r)" % (self.ra_0, self.dec_0, self.dec_1, self.dec_2)

class LambertConformalProjection(ConicProjection):
    def __init__(self, ra_0, dec_0, dec_1, dec_2):
        """Lambert Conformal conic projection.

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
        ConicProjection.__init__(self, ra_0, dec_0, dec_1, dec_2)
        # Snyder 1987, eq. 14-1, 14-2 and 15-1 to 15-3.
        self.dec_max = 89.99

        dec_1 *= self.deg2rad
        dec_2 *= self.deg2rad
        self.n = np.log(np.cos(dec_1)/np.cos(dec_2)) / \
        (np.log(np.tan(np.pi/4 + dec_2/2)/np.tan(np.pi/4 + dec_1/2)))
        self.F = np.cos(dec_1)*(np.tan(np.pi/4 + dec_1/2)**self.n)/self.n
        self.rho_0 = self._rho(dec_0)

    def _rho(self, dec):
        # check that dec is inside of -dec_max .. dec_max
        dec_ = np.array([dec], dtype='f8')
        dec_[dec_ < -self.dec_max] = -self.dec_max
        dec_[dec_ > self.dec_max] = self.dec_max
        return self.F / np.tan(np.pi/4 + dec_[0]/2 * self.deg2rad)**self.n

    def __call__(self, ra, dec, inverse=False):
        """Convert RA/Dec into map coordinates, or the reverse.

        Args:
            ra:  float or array of floats
            dec: float or array of floats
            inverse: if True, convert from map coordinates to RA/Dec

        Returns:
            x,y with the same format as ra/dec
        """
        if not inverse:
            ra_ = self._wrapRA(ra)
            theta = self.n * ra_
            rho = self._rho(dec)
            return rho*np.sin(theta * self.deg2rad), self.rho_0 - rho*np.cos(theta * self.deg2rad)
        else:
            # ra/dec actually x/y
            rho = np.sqrt(ra**2 + (self.rho_0 - dec)**2) * np.sign(self.n)
            if self.n >= 0:
                theta = np.arctan2(ra, self.rho_0 - dec) / self.deg2rad
            else:
                theta = np.arctan2(-ra, -(self.rho_0 - dec)) / self.deg2rad
            return self.ra_0 - theta/self.n, (2*np.arctan((self.F/rho)**(1./self.n)) - np.pi/2) / self.deg2rad

    def __repr__(self):
        return "LambertConformalProjection(%r, %r, %r, %r)" % (self.ra_0, self.dec_0, self.dec_1, self.dec_2)


class EquidistantProjection(ConicProjection):
    def __init__(self, ra_0, dec_0, dec_1, dec_2):
        """Equidistant conic projection.

        Equistant conic is a projection with an origin along the lines connecting
        the poles. It preserves distances along the map, but is not conformal,
        perspective or equal-area.

        Its preferred use of for smaller areas with predominant east-west extent
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
        ConicProjection.__init__(self, ra_0, dec_0, dec_1, dec_2)

        # Snyder 1987, eq. 14-3 to 14-6.
        self.n = (np.cos(dec_1 * self.deg2rad) - np.cos(dec_2 * self.deg2rad)) / (dec_2  - dec_1) / self.deg2rad
        self.G = np.cos(dec_1 * self.deg2rad)/self.n + (dec_1 * self.deg2rad)
        self.rho_0 = self._rho(dec_0)

    def _rho(self, dec):
        return self.G - (dec * self.deg2rad)

    def __call__(self, ra, dec, inverse=False):
        """Convert RA/Dec into map coordinates, or the reverse.

        Args:
            ra:  float or array of floats
            dec: float or array of floats
            inverse: if True, convert from map coordinates to RA/Dec

        Returns:
            x,y with the same format as ra/dec
        """
        if not inverse:
            ra_ = self._wrapRA(ra)
            # Snyder 1987, eq 16-1 to 16-4
            theta = self.n * ra_
            rho = self._rho(dec)
            return rho*np.sin(theta * self.deg2rad), self.rho_0 - rho*np.cos(theta * self.deg2rad)
        else:
            # ra/dec actually x/y
            # Snyder 1987, eq 14-10 to 14-11
            rho = np.sqrt(ra**2 + (self.rho_0 - dec)**2) * np.sign(self.n)
            if self.n >= 0:
                theta = np.arctan2(ra, self.rho_0 - dec) / self.deg2rad
            else:
                theta = np.arctan2(-ra, -(self.rho_0 - dec)) / self.deg2rad
            return self.ra_0 - theta/self.n, (self.G - rho)/ self.deg2rad

    def __repr__(self):
        return "EquidistantProjection(%r, %r, %r, %r)" % (self.ra_0, self.dec_0, self.dec_1, self.dec_2)

##### Start of free methods #####

def setMeridianPatches(ax, proj, meridians, **kwargs):
    """Add meridian lines to matplotlib axes.

    Meridian lines in conics are circular arcs, appropriate
    matplotlib.patches will be added to given ax.

    Args:
        ax: matplotlib axes
        proj: a projection class
        meridians: list of declinations
        **kwargs: matplotlib.patches.Arc parameters

    Returns:
        None
    """
    patches = proj.getMeridianPatches(meridians, **kwargs)
    ax.add_collection(patches)

def setParallelPatches(ax, proj, parallels, **kwargs):
    """Add parallel lines to matplotlib axes.

    Parallel lines in conics are straight, appropriate
    matplotlib.patches will be added to given ax.

    Args:
        ax: matplotlib axes
        proj: a projection class
        meridians: list of rectascensions
        **kwargs: matplotlib.collection.LineCollection parameters

    Returns:
        None
    """
    patches = proj.getParallelPatches(parallels, **kwargs)
    ax.add_collection(patches)

def degFormatter(deg):
    """Default formatter for map labels.

    Args:
        deg: float
    Returns:
        string
    """
    return "%d$^\circ$" % deg

def pmDegFormatter(deg):
    """String formatter for "+-%d^\circ"

    Args:
        deg: float

    Return:
        String
    """
    format = "%d$^\circ$"
    if deg > 0:
        format = "$+$" + format
    if deg < 0:
        format = "$-$" + format
    return format % np.abs(deg)

def hourAngleFormatter(ra):
    """String formatter for "hh:mm"

    Args:
        deg: float

    Return:
        String
    """
    if ra < 0:
        ra += 360
    hours = int(ra)/15
    minutes = int(float(ra - hours*15)/15 * 60)
    minutes = '{:>02}'.format(minutes)
    return "%d:%sh" % (hours, minutes)

def setMeridianLabels(ax, proj, meridians, loc="left", fmt=degFormatter, **kwargs):
    """Add labels for meridians to matplotlib axes.

    Args:
        ax: matplotlib axes
        proj: a projection class
        meridians: list of rectascensions
        loc: "left" or "right"
        fmt: string formatter for labels
        **kwargs: matplotlib tick parameters

    Returns:
        None
    """

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if loc == "left":
        ticks = []
        labels = []
        for m in meridians:
            tick = proj.findIntersectionAtX(xlim[0], ylim, dec=m)
            if tick is not None:
                ticks.append(tick)
                labels.append(fmt(m))
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, **kwargs)

    if loc == "right":
        ticks = []
        labels = []
        for m in meridians:
            tick = proj.findIntersectionAtX(xlim[1], ylim, dec=m)
            if tick is not None:
                ticks.append(tick)
                labels.append(fmt(m))
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, **kwargs)

    ax.set_ylim(ylim)

def setParallelLabels(ax, proj, parallels, loc="bottom", fmt=degFormatter, **kwargs):
    """Add labels for parallels to matplotlib axes.

    Args:
        ax: matplotlib axes
        proj: a projection class
        parallels: list of declinations
        loc: "top" or "bottom"
        fmt: string formatter for labels
        **kwargs: matplotlib tick parameters

    Returns:
        None
    """

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # remove duplicates
    parallels_ = np.unique(parallels % 360)

    if loc == "bottom":
        ticks = []
        labels = []
        for p in parallels_:
            p_ = p
            if p - proj.ra_0 < -180:
                p_ += 360
            if p - proj.ra_0 > 180:
                p_ -= 360
            tick = proj.findIntersectionAtY(ylim[0], xlim, ra=p_)
            if tick is not None:
                ticks.append(tick)
                labels.append(fmt(p))
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, **kwargs)

    if loc == "top":
        ticks = []
        labels = []
        for p in parallels_:
            p_ = p
            # center of map is at ra=ra_0: wrap it around
            if p - proj.ra_0 < -180:
                p_ += 360
            if p - proj.ra_0 > 180:
                p_ -= 360
            tick = proj.findIntersectionAtY(ylim[1], xlim, ra=p_)
            if tick is not None:
                ticks.append(tick)
                labels.append(fmt(p))
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, **kwargs)

    ax.set_xlim(xlim)

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

def setupConicAxes(ax, ra, dec, proj, pad=0.02):
    """Set up axes for conic projection.

    The function preconfigures the matplotlib axes and sets the proper x/y
    limits to show all of ra/dec.

    Args:
        ax: matplotlib axes
        ra: list of rectascensions
        dec: list of declinations
        proj: a projection instance
        pad: float, how much padding between data and map boundary

    Returns:
        None
    """
    # remove ticks as they look odd with curved/angled parallels/meridians
    ax.xaxis.set_tick_params(which='both', length=0)
    ax.yaxis.set_tick_params(which='both', length=0)

    # determine x/y limits
    x,y = proj(ra, dec)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    delta_xy = (xmax-xmin, ymax-ymin)
    xmin -= pad*delta_xy[0]
    xmax += pad*delta_xy[0]
    ymin -= pad*delta_xy[1]
    ymax += pad*delta_xy[1]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

def cloneMap(ax0, ax):
    """Convenience function to copy the setup of a map axes.

    Note that this sets up the axis, in particular the x/y limits, but does
    not clone any content (data or meridian/parellel patches or labels).

    Args:
        ax0: previousely configured matplotlib axes
        ax: axes to be configured

    Returns:
        None
    """
    ax.set_axis_bgcolor(ax0.get_axis_bgcolor())
    # remove ticks as they look odd with curved/angled parallels/meridians
    ax.xaxis.set_tick_params(which='both', length=0)
    ax.yaxis.set_tick_params(which='both', length=0)
    # set x/y limits
    ax.set_xlim(ax0.get_xlim())
    ax.set_ylim(ax0.get_ylim())

def createConicMap(ax, ra, dec, proj_class=None, ra0=None, dec0=None, pad=0.02, bgcolor='#aaaaaa'):
    """Create conic projection and set up axes.

    This function constructs a conic projection to optimally hold the
    ra/dec, see getOptimalConicProjection(),
    and  preconfigures the matplotlib axes and sets the proper x/y
    limits to show all of ra/dec.

    Args:
        ax: matplotlib axes
        ra: list of rectascensions
        dec: list of declinations
        proj: a projection instance, see getOptimalConicProjection()
        pad: float, how much padding between data and map boundary
        bgcolor: matplotlib color to be used for ax

    Returns:
        ConicProjection
    """

    proj = getOptimalConicProjection(ra, dec, proj_class=proj_class, ra0=ra0, dec0=dec0)
    setupConicAxes(ax, ra, dec, proj, pad=pad)
    return proj


def getHealpixVertices(pixels, nside, nest=False):
    import healpy as hp
    vertices = np.zeros((pixels.size, 4, 2))
    for i in xrange(pixels.size):
        corners = hp.vec2ang(np.transpose(hp.boundaries(nside,pixels[i], nest=nest)))
        corners = np.array(corners) * 180. / np.pi
        diff = corners[1] - corners[1][0]
        diff[diff > 180] -= 360
        diff[diff < -180] += 360
        corners[1] = corners[1][0] + diff
        vertices[i,:,0] = corners[1]
        vertices[i,:,1] = 90.0 - corners[0]
    return vertices

def getCountAtLocations(ra, dec, nside=512, per_area=True, return_vertices=False):
    """Get number density of objects from RA/Dec in HealPix cells.

    Requires: healpy

    Args:
        ra: list of rectascensions
        dec: list of declinations
        nside: HealPix nside
        per_area: return counts in units of 1/arcmin^2
        return_vertices: whether to also return the boundaries of HealPix cells

    Returns:
        bc, ra_, dec_, [vertices]
        bc: count of objects in a HealPix cell if count > 0
        ra_: rectascension of the cell center (same format as ra/dec)
        dec_: declinations of the cell center (same format as ra/dec)
        vertices: (N,4,2), RA/Dec coordinates of 4 boundary points of cell
    """
    import healpy as hp
    # get healpix pixels
    ipix = hp.ang2pix(nside, (90-dec)/180*np.pi, ra/180*np.pi, nest=False)
    # count how often each pixel is hit
    bc = np.bincount(ipix)
    pixels = np.nonzero(bc)[0]
    bc = bc[bc>0]
    if per_area:
        bc = bc.astype('f8')
        bc /= hp.nside2resol(nside, arcmin=True)**2 # in arcmin^-2
    # get position of each pixel in RA/Dec
    theta, phi = hp.pix2ang(nside, pixels, nest=False)
    ra_ = phi*180/np.pi
    dec_ = 90 - theta*180/np.pi

    # get the vertices that confine each pixel
    # convert to RA/Dec (thanks to Eric Huff)
    if return_vertices:
        vertices = getHealpixVertices(pixels, nside)
        return bc, ra_, dec_, vertices
    else:
        return bc, ra_, dec_

def reduceAtLocations(ra, dec, value, reduce_fct=np.mean, nside=512, return_vertices=False):
    """Reduce values at given RA/Dec in HealPix cells to a scalar.

    Requires: healpy

    Args:
        ra: list of rectascensions
        dec: list of declinations
        value: list of values to be reduced
        reduce_fct: function to operate on values
        nside: HealPix nside
        per_area: return counts in units of 1/arcmin^2
        return_vertices: whether to also return the boundaries of HealPix cells

    Returns:
        v, ra_, dec_, [vertices]
        v: reduction of values in a HealPix cell if count > 0
        ra_: rectascension of the cell center (same format as ra/dec)
        dec_: declinations of the cell center (same format as ra/dec)
        vertices: (N,4,2), RA/Dec coordinates of 4 boundary points of cell
    """
    import healpy as hp
    # get healpix pixels
    ipix = hp.ang2pix(nside, (90-dec)/180*np.pi, ra/180*np.pi, nest=False)
    # count how often each pixel is hit, only use non-empty pixels
    pixels = np.nonzero(np.bincount(ipix))[0]

    v = np.empty(pixels.size)
    for i in xrange(pixels.size):
        sel = (ipix == pixels[i])
        v[i] = reduce_fct(value[sel])

    # get position of each pixel in RA/Dec
    theta, phi = hp.pix2ang(nside, pixels, nest=False)
    ra_ = phi*180/np.pi
    dec_ = 90 - theta*180/np.pi

    # get the vertices that confine each pixel
    # convert to RA/Dec (thanks to Eric Huff)
    if return_vertices:
        vertices = getHealpixVertices(pixels, nside)
        return v, ra_, dec_, vertices
    else:
        return v, ra_, dec_


def createFigureAx(ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
    else:
        fig = ax.get_figure()
    return fig, ax


def plotDensity(ra, dec, nside=1024, sep=5, cmap="YlOrRd", bgcolor="#aaaaaa", colorbar=True, cb_label='$n$ [arcmin$^{-2}$]', proj_class=None, ax=None):
    """Plot density map on optimally chosen projection.

    Args:
        ra: list of rectascensions
        dec: list of declinations
        nside: HealPix nside
        sep: separation of graticules [deg]
        cmap: colormap name
        bgcolor: background color of ax
        colorbar: whether to draw colorbar
        cb_label: label of colorbar
        proj_class: constructor of projection class, see getOptimalConicProjection()
        ax: matplotlib axes (will be created if not given)
    Returns:
        figure, axes, projection
    """

    # setup figure
    fig, ax = createFigureAx(ax=ax)

    # setup map: define map optimal for given RA/Dec
    proj = createConicMap(ax, ra, dec, proj_class=proj_class)

    # get count in healpix cells, restrict to non-empty cells
    bc, _, _, vertices = getCountAtLocations(ra, dec, nside=nside, return_vertices=True)

    # make a map of the vertices
    poly = makeVertexMap(vertices, bc, proj, ax, cmap=cmap)

    # do we want colorbar?
    if not colorbar:
        poly = None

    # create nice map
    makeMapNice(fig, ax, proj, dec, sep=sep, bgcolor=bgcolor, cb_collection=poly, cb_label=cb_label)

    fig.show()
    return fig, ax, proj


def plotHealpix(m, nside, nest=False, use_vertices=True, sep=5, cmap="YlOrRd", bgcolor="#aaaaaa", colorbar=True, cb_label="Healpix value", proj_class=None, ax=None):
    """Plot HealPix map on optimally chosen projection.

    Args:
        m: Healpix map array
        nside: HealPix nside
        nest: HealPix nest
        use_vertices: calculate individual polygons per HealPix cell
        sep: separation of graticules [deg]
        cmap: colormap name
        bgcolor: background color of ax
        colorbar: whether to draw colorbar
        cb_label: label of colorbar
        proj_class: constructor of projection class, see getOptimalConicProjection()
        ax: matplotlib axes (will be created if not given)
    Returns:
        figure, axes, projection
    """

    # setup figure
    fig, ax = createFigureAx(ax=ax)

    # determine ra, dec of map; restrict to non-empty cells
    pixels = np.flatnonzero(m)

    vertices = getHealpixVertices(pixels, nside, nest=nest)
    ra_dec = vertices.mean(axis=1)
    ra, dec = ra_dec[:,0], ra_dec[:,1]

    # setup map: define map optimal for given RA/Dec
    proj = createConicMap(ax, ra, dec, proj_class=proj_class)

    # make a map of the vertices
    if use_vertices:
        poly = makeVertexMap(vertices, m[pixels], proj, ax, cmap=cmap)
    else:
        poly = makeScatterMap(ra, dec, m[pixels], proj, ax, cmap=cmap)

    # do we want colorbar?
    if not colorbar:
        poly = None

    # create nice map
    makeMapNice(fig, ax, proj, dec, sep=sep, bgcolor=bgcolor, cb_collection=poly, cb_label=cb_label)

    fig.show()
    return fig, ax, proj


def plotMap(ra, dec, value, sep=5, marker="h", markersize=None, cmap="YlOrRd", bgcolor="#aaaaaa", colorbar=True, cb_label="Map value", proj_class=None, ax=None):
    """Plot map values on optimally chosen projection.

    Args:
        ra: list of rectascensions
        dec: list of declinations
        value: list of map values
        sep: separation of graticules [deg]
        marker: matplotlib marker name (e.g. 's','h','o')
        markersize: size of marker (in points^2), uses best guess if not set
        cmap: colormap name
        bgcolor: background color of ax
        colorbar: whether to draw colorbar
        cb_label: label of colorbar
        proj_class: constructor of projection class, see getOptimalConicProjection()
        ax: matplotlib axes (will be created if not given)
    Returns:
        figure, axes, projection
    """

    # setup figure
    fig, ax = createFigureAx(ax=ax)

    # setup map: define map optimal for given RA/Dec
    proj = createConicMap(ax, ra, dec, proj_class=proj_class)

    # make a map of the ra/dec/value points
    sc = makeScatterMap(ra, dec, value, proj, ax, marker=marker, markersize=markersize, cmap=cmap)

    # do we want colorbar?
    if not colorbar:
        sc = None

    # create nice map
    makeMapNice(fig, ax, proj, dec, sep=sep, bgcolor=bgcolor, cb_collection=sc, cb_label=cb_label)

    fig.show()
    return fig, ax, proj


def makeVertexMap(vertices, color, proj, ax, cmap="YlOrRd"):
    # add healpix counts from vertices
    vmin, vmax = np.percentile(color,[10,90])
    return addPolygons(vertices, proj, ax, color=color, vmin=vmin, vmax=vmax, cmap=cmap, zorder=3, rasterized=True)

def makeScatterMap(ra, dec, val, proj, ax, marker="s", markersize=None, cmap="YlOrRd"):
    x,y = proj(ra, dec)
    fig = ax.get_figure()
    if markersize is None:
        markersize = getMarkerSizeToFill(fig, ax, x, y)
    vmin, vmax = np.percentile(val,[10,90])
    sc = ax.scatter(x, y, c=val, marker=marker, s=markersize, edgecolors='None', zorder=3, vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
    return sc

def makeMapNice(fig, ax, proj, dec, sep=5, bgcolor="#aaaaaa", cb_collection=None, cb_label=""):
    # add lines and labels for meridians/parallels
    meridians = np.arange(-90, 90+sep, sep)
    parallels = np.arange(0, 360+sep, sep)
    setMeridianPatches(ax, proj, meridians, linestyle='-', lw=0.5, alpha=0.2, zorder=2)
    setParallelPatches(ax, proj, parallels, linestyle='-', lw=0.5, alpha=0.2, zorder=2)
    setMeridianLabels(ax, proj, meridians, loc="left", fmt=pmDegFormatter)
    if dec.mean() > 0:
        setParallelLabels(ax, proj, parallels, loc="bottom")
    else:
        setParallelLabels(ax, proj, parallels, loc="top")

    if bgcolor is not None:
        ax.set_facecolor(bgcolor)

    # add colorbar
    if cb_collection is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.0)
        cb = fig.colorbar(cb_collection, cax=cax)
        cb.set_label(cb_label)
        cb.solids.set_edgecolor("face")
    fig.tight_layout()


# decorator for registering the survey footprint loader functions
footprint_loader = {}

def register(surveyname=""):
    def decorate(func):
        footprint_loader[surveyname] = func
        return func
    return decorate


def addFootprint(surveyname, proj, ax, **kwargs):
    """Plot survey footprint polygon onto map.

    Args:
        surveyname: name of the survey
        proj: map projection
        ax: matplotlib axes
        **kwargs: matplotlib.collections.PolyCollection keywords
    Returns:
        figure, axes, matplotlib.collections.PolyCollection
    """

    # setup figure
    if ax is None:
        fig, ax = createFigureAx()
    else:
        fig = ax.get_figure()

    ra, dec = footprint_loader[surveyname]()
    x,y  = proj(ra, dec)
    from matplotlib.patches import Polygon
    poly = Polygon(np.dstack((x,y))[0], closed=True, **kwargs)
    ax.add_artist(poly)
    return fig, ax, poly


def addPolygons(vertices, proj, ax, color=None, vmin=None, vmax=None, **kwargs):
    """Plot polygons (e.g. Healpix cells) onto map.

    Args:
        vertices: Healpix cell boundaries in RA/Dec, from getCountAtLocations()
        proj: map projection
        ax: matplotlib axes
        color: string or matplib color, or numeric array to set polygon colors
        vmin: if color is numeric array, use vmin to set color of minimum
        vmax: if color is numeric array, use vmin to set color of minimum
        **kwargs: matplotlib.collections.PolyCollection keywords
    Returns:
        matplotlib.collections.PolyCollection
    """
    from matplotlib.collections import PolyCollection
    vertices_ = np.empty_like(vertices)
    vertices_[:,:,0], vertices_[:,:,1] = proj(vertices[:,:,0], vertices[:,:,1])
    coll = PolyCollection(vertices_, array=color, **kwargs)
    coll.set_clim(vmin=vmin, vmax=vmax)
    coll.set_edgecolor("face")
    ax.add_collection(coll)
    return coll


def getMarkerSizeToFill(fig, ax, x, y):
    """Get the size of a marker so that data points can fill axes.

    Assuming that x/y span a rectangle inside of ax, this method computes
    a best guess of the marker size to completely fill the area.

    Note: The marker area calculation in matplotlib seems to assume a square
          shape. If others shapes are desired (e.g. 'h'), a mild increase in
          size will be necessary.

    Args:
        fig: matplotlib.figure
        ax: matplib.axes that should hold x,y
        x, y: list of map positions

    Returns:
        int, the size (actually: area) to be used for scatter(..., s= )
    """
    # get size of bounding box in pixels
    # from http://stackoverflow.com/questions/19306510/determine-matplotlib-axis-size-in-pixels
    from math import ceil
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    dx = x.max() - x.min()
    dy = y.max() - y.min()
    filling_x = dx / (xlim[1] - xlim[0])
    filling_y = dy / (ylim[1] - ylim[0])
    # assuming x,y to ~fill a rectangle: get the point density
    area = filling_x*filling_y * width * height
    s = area / x.size
    return int(ceil(s)) # round up to be on the safe side
