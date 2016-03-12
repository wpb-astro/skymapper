import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

class AlbersEqualAreaProjection(object):
    def __init__(self, ra_0, dec_0, dec_1, dec_2):
        self.ra_0 = ra_0
        self.dec_0 = dec_0
        self.dec_1 = dec_1 # dec1 and dec2 only needed for __repr__
        self.dec_2 = dec_2
        self.deg2rad = np.pi/180

        self.n = (np.sin(dec_1 * self.deg2rad) + np.sin(dec_2 * self.deg2rad)) / 2
        self.C = np.cos(dec_1 * self.deg2rad)**2 + 2 * self.n * np.sin(dec_1 * self.deg2rad)
        self.rho_0 = self.__rho__(dec_0)

    def __rho__(self, dec):
        return np.sqrt(self.C - 2 * self.n * np.sin(dec * self.deg2rad)) / self.n

    def __call__(self, ra, dec, inverse=False):
        if not inverse:

            ra_ = np.array([ra - self.ra_0]) * -1 # inverse for RA
            # check that ra_ is between -180 and 180 deg
            ra_[ra_ < -180 ] += 360
            ra_[ra_ > 180 ] -= 360

            theta = self.n * ra_[0]
            rho = self.__rho__(dec)
            return rho*np.sin(theta * self.deg2rad), self.rho_0 - rho*np.cos(theta * self.deg2rad)
        else:
            # ra/dec actually x/y
            rho = np.sqrt(ra**2 + (self.rho_0 - dec)**2)
            theta = np.arctan(ra/(self.rho_0 - dec)) / self.deg2rad
            return self.ra_0 - theta/self.n, np.arcsin((self.C - (rho * self.n)**2)/(2*self.n)) / self.deg2rad

    def __repr__(self):
        return "AlbersEqualAreaProjection(%r, %r, %r, %r)" % (self.ra_0, self.dec_0, self.dec_1, self.dec_2)

    def getMeridianPatches(self, meridians, **kwargs):
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
        return PatchCollection(patches, match_original=True)

    def getParallelPatches(self, parallels, **kwargs):
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
        from scipy.optimize import newton
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
            try:
                return newton(lambda y: self.__call__(x,y,inverse=True)[0] - ra, (ylim[0] + ylim[1])/2)
            except RuntimeError:
                return None
        raise NotImplementedError("specify either RA or Dec")

    def findIntersectionAtY(self, y, xlim, ra=None, dec=None):
        from scipy.optimize import newton
        if dec is not None:
            # analytic solution for intersection of circle with line at x
            r = np.abs(self.rho_0 - self.__call__(self.ra_0, dec)[1])
            if np.abs(y) > np.abs(r):
                return None
            if self.rho_0 >= 0:
                return self.rho_0 - np.sqrt(r**2 - y**2)
            else:
                return np.sqrt(r**2 - y**2) + self.rho_0
        if ra is not None:
            try:
                return newton(lambda x: self.__call__(x,y,inverse=True)[0] - ra, (xlim[0] + xlim[1])/2)
            except RuntimeError:
                return None
        raise NotImplementedError("specify either RA or Dec")

    def degFormatter(deg):
        return "%d$^\circ$" % deg

    def setMeridianLabels(self, ax, meridians, loc="left", fmt=degFormatter):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if loc == "left":
            ticks = []
            labels = []
            for m in meridians:
                tick = self.findIntersectionAtX(xlim[0], ylim, dec=m)
                if tick is not None:
                    ticks.append(tick)
                    labels.append(fmt(m))
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)

        if loc == "right":
            ticks = []
            labels = []
            for m in meridians:
                tick = self.findIntersectionAtX(xlim[1], ylim, dec=m)
                if tick is not None:
                    ticks.append(tick)
                    labels.append(fmt(m))
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)

        ax.set_ylim(ylim)

    def setParallelLabels(self, ax, parallels, loc="bottom", fmt=degFormatter):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # remove duplicates
        parallels_ = np.unique(parallels % 360)

        # the outer boundaries need to be duplicated because the same
        # parallel appears on the left and the right side of the map
        if self.ra_0 < 180:
            outer = self.ra_0 - 180
        else:
            outer = self.ra_0 + 180
        parallels_ = np.array(list(parallels_) + [outer])

        if loc == "bottom":
            ticks = []
            labels = []
            for p in parallels_:
                p_ = p
                if p - self.ra_0 < -180:
                    p_ += 360
                if p - self.ra_0 > 180:
                    p_ -= 360
                tick = self.findIntersectionAtY(ylim[0], xlim, ra=p_)
                if tick is not None:
                    ticks.append(tick)
                    labels.append(fmt(p))
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)

        if loc == "top":
            ticks = []
            labels = []
            for p in parallels_:
                p_ = p
                # center of map is at ra=ra_0: wrap it around
                if p - self.ra_0 < -180:
                    p_ += 360
                if p - self.ra_0 > 180:
                    p_ -= 360
                tick = self.findIntersectionAtY(ylim[1], xlim, ra=p_)
                if tick is not None:
                    ticks.append(tick)
                    labels.append(fmt(p))
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)

        ax.set_xlim(xlim)

def pmDegFormatter(deg):
    format = "%d$^\circ$"
    if deg > 0:
        format = "$+$" + format
    if deg < 0:
        format = "$-$" + format
    return format % np.abs(deg)

def hourAngleFormatter(ra):
    if ra < 0:
        ra += 360
    hours = int(ra)/15
    minutes = int(float(ra - hours*15)/15 * 60)
    minutes = '{:>02}'.format(minutes)
    return "%d:%sh" % (hours, minutes)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
aea = AlbersEqualAreaProjection(60, 0., -20, 10)
meridians = np.linspace(-90, 90, 13)
parallels = np.linspace(0, 360, 25)
patches = aea.getMeridianPatches(meridians, linestyle=':', lw=0.5)
ax.add_collection(patches)
patches = aea.getParallelPatches(parallels, linestyle=':', lw=0.5)
ax.add_collection(patches)
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.8, 0.8)
aea.setMeridianLabels(ax, meridians, loc="left", fmt=pmDegFormatter)
aea.setParallelLabels(ax, parallels, loc="bottom", fmt=hourAngleFormatter)
plt.tick_params(which='both', length=0)
plt.show()
