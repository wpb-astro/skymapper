import matplotlib
import numpy as np
import re
from . import healpix

# decorator for registering the survey footprint loader functions
footprint_loader = {}

def register(surveyname=""):
    def decorate(func):
        footprint_loader[surveyname] = func
        return func
    return decorate

DEG2RAD = np.pi/180
resolution = 75

# extrapolation function from
# http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range
# improved to order x and y to have ascending x
def extrap(x, xp, yp):
    """np.interp function with linear extrapolation"""
    x_ = np.array(x)
    order = np.argsort(xp)
    xp_ = xp[order]
    yp_ = yp[order]

    y = np.array(np.interp(x_, xp_, yp_))
    y[x_ < xp_[0]] = yp_[0] + (x_[x_ < xp_[0]] -xp_[0]) * (yp_[0] - yp_[1]) / (xp_[0] - xp_[1])
    y[x_ > xp_[-1]] = yp_[-1] + (x_[x_ > xp_[-1]] -xp_[-1])*(yp_[-1] - yp_[-2])/(xp_[-1] - xp_[-2])
    return y

def degFormatter(deg):
    """Default formatter for map labels.

    Args:
        deg: float
    Returns:
        string
    """
    return "$%d^\circ$" % deg

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


class Map():
    def __init__(self, proj, ax=None, interactive=True, **kwargs):
        self.proj = proj
        self._setFigureAx(ax, interactive=interactive)
        self._setEdge(**kwargs)

    def _setFigureAx(self, ax=None, interactive=True):
        if ax is None:
            self.fig = matplotlib.pyplot.figure()
            self.ax = self.fig.add_subplot(111, aspect='equal')
        else:
            self.ax = ax
            self.ax.set_aspect('equal')
            self.fig = self.ax.get_figure()
        self.ax.set_axis_off()
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])

        # attach event handlers
        if interactive:
            self._set_frame_args = {}
            self._set_meridianlabelframe_args = {}
            self._set_parallellabelframe_args = {}
            self._press_evt = self.fig.canvas.mpl_connect('button_press_event', self._pressHandler)
            self._release_evt = self.fig.canvas.mpl_connect('button_release_event', self._releaseHandler)
            self._scroll_evt = self.fig.canvas.mpl_connect('scroll_event', self._scrollHandler)

    @property
    def parallels(self):
        return [ float(m.group(1)) for c,m in self.artists(r'grid-parallel-([\-\+0-9.]+)', regex=True) ]

    @property
    def meridians(self):
        return [ float(m.group(1)) for c,m in self.artists(r'grid-meridian-([\-\+0-9.]+)', regex=True) ]

    def artists(self, gid, regex=False):
        if regex:
            matches = [ re.match(gid, c.get_gid()) if c.get_gid() is not None else None for c in self.ax.get_children() ]
            return [ (c,m) for c,m in zip(self.ax.get_children(), matches) if m is not None ]
        else: # direct match
            return [ c for c in self.ax.get_children() if c.get_gid() is not None and c.get_gid().find(gid) != -1 ]

    def _setParallel(self, p, **kwargs):
        ls = kwargs.pop('ls', '-')
        lw = kwargs.pop('lw', 0.5)
        c = kwargs.pop('c', 'k')
        alpha = kwargs.pop('alpha', 0.2)
        zorder = kwargs.pop('zorder', 10)
        x_, y_ = self.proj.transform(self._ra_range, p*np.ones(len(self._ra_range)))
        self.ax.plot(x_, y_, ls=ls, lw=lw, c=c, alpha=alpha, zorder=zorder, **kwargs)

    def _setMeridian(self, m, **kwargs):
        ls = kwargs.pop('ls', '-')
        lw = kwargs.pop('lw', 0.5)
        c = kwargs.pop('c', 'k')
        alpha = kwargs.pop('alpha', 0.2)
        zorder = kwargs.pop('zorder', 10)
        x_, y_ = self.proj.transform(m*np.ones(len(self._dec_range)), self._dec_range)
        self.ax.plot(x_, y_, ls=ls, lw=lw, c=c, alpha=alpha, zorder=zorder, **kwargs)

    def _setEdge(self, **kwargs):
        self._dec_range = np.linspace(-90, 90, resolution)
        self._ra_range = np.linspace(-180, 180, resolution) + self.proj.ra_0

        lw = kwargs.pop('lw', 1)
        c = kwargs.pop('c', '#444444')
        alpha = kwargs.pop('alpha', 1)
        zorder = kwargs.pop('zorder', 100)

        for p in [-90, 90]:
            self._setParallel(p, lw=lw, c=c, alpha=alpha, zorder=zorder, gid='edge-parallel', **kwargs)
        for m in [self.proj.ra_0 + 180, self.proj.ra_0 - 180]:
            self._setMeridian(m, lw=lw, c=c, alpha=alpha, zorder=zorder, gid='edge-meridian', **kwargs)

    def grid(self, sep=30, parallel_fmt=pmDegFormatter, meridian_fmt=degFormatter, dec_min=-90, dec_max=90, ra_min=-180, ra_max=180, **kwargs):
        self.parallel_fmt = parallel_fmt
        self.meridian_fmt = meridian_fmt
        self._dec_range = np.linspace(dec_min, dec_max, resolution)
        self._ra_range = np.linspace(ra_min, ra_max, resolution) + self.proj.ra_0
        _parallels = np.arange(-90+sep,90,sep)
        if self.proj.ra_0 % sep == 0:
            _meridians = np.arange(sep * ((self.proj.ra_0 + 180) // sep), sep * ((self.proj.ra_0 - 180) // sep - 1), -sep)
        else:
            _meridians = np.arange(sep * ((self.proj.ra_0 + 180) // sep), sep * ((self.proj.ra_0 - 180) // sep), -sep)

        # clean up previous grid
        artists = self.artists('grid-meridian') + self.artists('grid-parallel')
        for artist in artists:
                artist.remove()

        # clean up meridian and parallel labels because they're tied to the grid
        artists = self.artists('meridian-label') + self.artists('parallel-label')
        for artist in artists:
                artist.remove()

        for p in _parallels:
            self._setParallel(p, gid='grid-parallel-%r' % p, **kwargs)
        for m in _meridians:
            self._setMeridian(m, gid='grid-meridian-%r' % m, **kwargs)

    def gradient(self, ra, dec, sep=1e-2, direction='parallel'):
        # gradients in *positive* dec and *negative* ra
        assert direction in ['parallel', 'meridian']
        correction = 1
        if direction == 'parallel':
            testm = np.array([ra+sep/2, ra-sep/2])
            if testm[0] >= self.proj.ra_0 + 180:
                testm[0] = ra
                correction = 2
            if testm[1] <= self.proj.ra_0 - 180:
                testm[1] = ra
                correction = 2
            x_, y_ = self.proj.transform(testm, np.ones(2)*dec)
        else:
            testp = np.array([dec-sep/2, dec+sep/2])
            if testp[0] <= -90:
                testp[0] = dec
                correction = 2
            if testp[1] >= 90:
                testp[1] = dec
                correction = 2
            x_, y_ = self.proj.transform(np.ones(2)*ra, testp)
        return np.array((x_[1] - x_[0], y_[1] - y_[0])) * correction

    def _negateLoc(self, loc):
        if loc == "bottom":
            return "top"
        if loc == "top":
            return "bottom"
        if loc == "left":
            return "right"
        if loc == "right":
            return "left"

    def labelMeridianAtParallel(self, p, loc=None, meridians=None, pad=None, direction='parallel', **kwargs):

        if loc is None:
            if p >= 0:
                loc = 'top'
            else:
                loc = 'bottom'
        assert loc in ['top', 'bottom']

        horizontalalignment = kwargs.pop('horizontalalignment', 'center')
        verticalalignment = kwargs.pop('verticalalignment', self._negateLoc(loc))
        zorder = kwargs.pop('zorder', 20)
        rotation = kwargs.pop('rotation', None)
        size = kwargs.pop('size', matplotlib.rcParams['font.size'])
        if pad is None:
            pad = size / 3

        # determine rot_base so that central label is upright
        if rotation is None:
            m = self.proj.ra_0
            dxy = self.gradient(m, p, direction=direction)
            angle = np.arctan2(dxy[0], dxy[1]) / DEG2RAD
            options = np.arange(-2,3) * 90 # multiples of 90 deg
            closest = np.argmin(np.abs(options - angle))
            rot_base = options[closest]

        if meridians is None:
            meridians = self.meridians

        for m in meridians:
            # move label along meridian
            xp, yp = self.proj.transform(m, p)
            dxy = self.gradient(m, p, direction="meridian")
            dxy *= pad / np.sqrt((dxy**2).sum())
            if loc == 'bottom':
                dxy *= -1

            if rotation is None:
                dxy_ = self.gradient(m, p, direction=direction)
                angle = rot_base-np.arctan2(dxy_[0], dxy_[1]) / DEG2RAD
            else:
                angle = rotation

            if m < 0:
                m += 360

            self.ax.annotate(self.meridian_fmt(m), (xp, yp), xytext=dxy, textcoords='offset points', rotation=angle, rotation_mode='anchor', horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, size=size, zorder=zorder, gid='meridian-label', **kwargs)

    def labelParallelAtMeridian(self, m, loc=None, parallels=None, pad=None, direction='parallel', **kwargs):

        if loc is None:
            if m <= 0:
                loc = 'right'
            else:
                loc = 'left'
        assert loc in ['left', 'right']

        horizontalalignment = kwargs.pop('horizontalalignment', self._negateLoc(loc))
        verticalalignment = kwargs.pop('verticalalignment', 'center')
        zorder = kwargs.pop('zorder', 20)
        rotation = kwargs.pop('rotation', None)
        size = kwargs.pop('size', matplotlib.rcParams['font.size'])
        if pad is None:
            pad = size / 3

        # determine rot_base so that central label is upright
        if rotation is None:
            p = 0
            dxy = self.gradient(m, p, direction=direction)
            angle = np.arctan2(dxy[0], dxy[1]) / DEG2RAD
            options = np.arange(-2,3) * 90
            closest = np.argmin(np.abs(options - angle))
            rot_base = options[closest]

        if parallels is None:
            parallels = self.parallels

        for p in parallels:
            # move label along parallel
            xp, yp = self.proj.transform(m, p)
            dxy = self.gradient(m, p, direction="parallel")
            dxy *= pad / np.sqrt((dxy**2).sum())
            if loc == 'left':
                dxy *= -1

            if rotation is None:
                dxy_ = self.gradient(m, p, direction=direction)
                angle = rot_base-np.arctan2(dxy_[0], dxy_[1]) / DEG2RAD
            else:
                angle = rotation

            self.ax.annotate(self.parallel_fmt(p), (xp, yp), xytext=dxy, textcoords='offset points', rotation=angle, rotation_mode='anchor',  horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, size=size, zorder=zorder,  gid='parallel-label', **kwargs)

    def labelMeridiansAtFrame(self, loc=None, meridians=None, pad=None, **kwargs):
        self._set_meridianlabelframe_args = locals()
        self._set_meridianlabelframe_args.pop('self')
        for k,v in self._set_meridianlabelframe_args.pop('kwargs'):
            self._set_meridianlabelframe_args[k]=v

        locs = ['top', 'bottom']
        if loc is not None:
            assert loc in locs
            locs = [loc]

        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        horizontalalignment = kwargs.pop('horizontalalignment', 'center')
        _ = kwargs.pop('verticalalignment', None) # no option along the frame
        size = kwargs.pop('size', matplotlib.rcParams['font.size'])
        if pad is None:
            pad = size / 3

        if meridians is None:
            meridians = self.meridians

        poss = {"bottom": 0, "top": 1}

        # check if loc has frame
        frame_artists = self.artists(r'frame-([a-zA-Z]+)', regex=True)
        frame_locs = [match.group(1) for c,match in frame_artists]
        for loc in locs:
            pos = poss[loc]
            zorder = kwargs.pop('zorder', self.ax.spines[loc].get_zorder())
            verticalalignment = self._negateLoc(loc) # no option along the frame

            if loc in frame_locs:
                # find all parallel grid lines
                m_artists = self.artists(r'grid-meridian-([\-\+0-9.]+)', regex=True)
                for c,match in m_artists:
                    m = float(match.group(1))
                    if m in meridians:
                        # intersect with axis
                        xm, ym = c.get_xdata(), c.get_ydata()
                        xm_at_ylim = extrap(ylim, ym, xm)[pos]
                        if xm_at_ylim >= xlim[0] and xm_at_ylim <= xlim[1] and self.proj.contains(xm_at_ylim, ylim[pos]):
                            m_, p_ = self.proj.invert(xm_at_ylim, ylim[pos])
                            dxy = self.gradient(m_, p_, direction="meridian")
                            dxy /= np.sqrt((dxy**2).sum())
                            dxy *= pad / dxy[1] # same pad from frame
                            if loc == "bottom":
                                dxy *= -1
                            angle = 0 # no option along the frame

                            x_im = (xm_at_ylim - xlim[0])/(xlim[1]-xlim[0])
                            y_im = (ylim[pos] - ylim[0])/(ylim[1]-ylim[0])

                            if m < 0:
                                m += 360

                            self.ax.annotate(self.meridian_fmt(m), (x_im, y_im), xycoords='axes fraction', xytext=dxy, textcoords='offset points', annotation_clip=False,  gid='frame-meridian-label', horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, size=size, zorder=zorder,  **kwargs)

    def labelParallelsAtFrame(self, loc=None, parallels=None, pad=None, **kwargs):

        self._set_parallellabelframe_args = locals()
        self._set_parallellabelframe_args.pop('self')
        for k,v in self._set_parallellabelframe_args.pop('kwargs'):
            self._set_parallellabelframe_args[k]=v

        locs = ['left', 'right']
        if loc is not None:
            assert loc in locs
            locs = [loc]

        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()

        size = kwargs.pop('size', matplotlib.rcParams['font.size'])
        verticalalignment = kwargs.pop('verticalalignment', 'center')
        _ = kwargs.pop('horizontalalignment', None) # no option along the frame

        if pad is None:
            pad = size / 3

        if parallels is None:
            parallels = self.parallels

        poss = {"left": 0, "right": 1}

        # check if loc has frame
        frame_artists = self.artists(r'frame-([a-zA-Z]+)', regex=True)
        frame_locs = [match.group(1) for c,match in frame_artists]
        for loc in locs:
            pos = poss[loc]
            zorder = kwargs.pop('zorder', self.ax.spines[loc].get_zorder())
            horizontalalignment = self._negateLoc(loc) # no option along the frame

            if loc in frame_locs:
                # find all parallel grid lines
                m_artists = self.artists(r'grid-parallel-([\-\+0-9.]+)', regex=True)
                for c,match in m_artists:
                    p = float(match.group(1))
                    if p in parallels:
                        # intersect with axis
                        xp, yp = c.get_xdata(), c.get_ydata()
                        yp_at_xlim = extrap(xlim, xp, yp)[pos]
                        if yp_at_xlim >= ylim[0] and yp_at_xlim <= ylim[1] and self.proj.contains(xlim[pos], yp_at_xlim):
                            m_, p_ = self.proj.invert(xlim[pos], yp_at_xlim)
                            dxy = self.gradient(m_, p_, direction='parallel')
                            dxy /= np.sqrt((dxy**2).sum())
                            dxy *= pad / dxy[0] # same pad from frame
                            if loc == "left":
                                dxy *= -1
                            angle = 0 # no option along the frame

                            x_im = (xlim[pos] - xlim[0])/(xlim[1]-xlim[0])
                            y_im = (yp_at_xlim - ylim[0])/(ylim[1]-ylim[0])
                            self.ax.annotate(self.parallel_fmt(p), (x_im, y_im), xycoords='axes fraction', xytext=dxy, textcoords='offset points', annotation_clip=False, gid='frame-parallel-label', horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, size=size, zorder=zorder,  **kwargs)


    def _setFrame(self, precision=1000):
        # remember function arguments to recreate
        self._set_frame_args = locals()
        self._set_frame_args.pop('self')

        # clean up existing frame
        frame_artists = self.artists(r'frame-([a-zA-Z]+)', regex=True)
        for c,m in frame_artists:
            c.remove()

        locs = ['left', 'bottom', 'right', 'top']
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()

        for loc in locs:
            # define line along axis
            const = np.ones(precision)
            if loc == "left":
                line = xlim[0]*const, np.linspace(ylim[0], ylim[1], precision)
            if loc == "right":
                line = xlim[1]*const, np.linspace(ylim[0], ylim[1], precision)
            if loc == "bottom":
                line = np.linspace(xlim[0], xlim[1], precision), ylim[0]*const
            if loc == "top":
                line = np.linspace(xlim[0], xlim[1], precision), ylim[1]*const

            # use styling of spine to mimic axes
            ls = self.ax.spines[loc].get_ls()
            lw = self.ax.spines[loc].get_lw()
            c = self.ax.spines[loc].get_edgecolor()
            alpha = self.ax.spines[loc].get_alpha()
            zorder = self.ax.spines[loc].get_zorder()

            # show axis lines only where line is inside of map edge
            inside = self.proj.contains(*line)
            if (~inside).all():
                continue

            if inside.all():
                startpos, stoppos = 0, -1
                xmin = (line[0][startpos] - xlim[0])/(xlim[1]-xlim[0])
                ymin = (line[1][startpos] - ylim[0])/(ylim[1]-ylim[0])
                xmax = (line[0][stoppos] - xlim[0])/(xlim[1]-xlim[0])
                ymax = (line[1][stoppos] - ylim[0])/(ylim[1]-ylim[0])
                self.ax.plot([xmin,xmax], [ymin, ymax], c=c, ls=ls, lw=lw, alpha=alpha, zorder=zorder, clip_on=False, transform=self.ax.transAxes, gid='frame-%s' % loc)
                continue

            # for piecewise inside: determine limits where it's inside
            # by checking for jumps in inside
            inside = inside.astype("int")
            diff = inside[1:] - inside[:-1]
            jump = np.flatnonzero(diff)
            start = 0
            if inside[0]:
                jump = np.concatenate(((0,),jump))

            while True:
                startpos = jump[start]
                if start+1 < len(jump):
                    stoppos = jump[start + 1]
                else:
                    stoppos = -1

                xmin = (line[0][startpos] - xlim[0])/(xlim[1]-xlim[0])
                ymin = (line[1][startpos] - ylim[0])/(ylim[1]-ylim[0])
                xmax = (line[0][stoppos] - xlim[0])/(xlim[1]-xlim[0])
                ymax = (line[1][stoppos] - ylim[0])/(ylim[1]-ylim[0])
                self.ax.plot([xmin,xmax], [ymin, ymax], c=c, ls=ls, lw=lw, alpha=alpha, zorder=zorder, clip_on=False, transform=self.ax.transAxes, gid='frame-%s' % loc)
                if start + 2 < len(jump):
                    start += 2
                else:
                    break

    def _clearFrame(self):
        frame_artists = self.artists('frame-')
        for artist in frame_artists:
            artist.remove()

    def _resetFrame(self):
        self._setFrame(**self._set_frame_args)
        self.labelMeridiansAtFrame(**self._set_meridianlabelframe_args)
        self.labelParallelsAtFrame(**self._set_parallellabelframe_args)

    def _pressHandler(self, evt):
        if evt.button != 1: return
        if evt.dblclick: return
        # remove frame and labels
        self._clearFrame()
        self.fig.canvas.draw()

    def _scrollHandler(self, evt):
        # mouse scroll for zoom
        if evt.inaxes != self.ax: return
        if evt.step == 0: return

        # remove frame and labels
        self._clearFrame()

        # scroll to fixed pointer position: google maps style
        factor = 0.25
        c = 1 - evt.step*factor # scaling factor
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        xdiff, ydiff = xlim[1] - xlim[0], ylim[1] - ylim[0]
        x, y = evt.xdata, evt.ydata
        fx, fy = (x - xlim[0])/xdiff, (y - ylim[0])/ydiff # axis units
        xlim_, ylim_ = x - fx*c*xdiff, y - fy*c*ydiff
        xlim__, ylim__ = xlim_ + c*xdiff, ylim_ + c*ydiff

        self.ax.set_xlim(xlim_, xlim__)
        self.ax.set_ylim(ylim_, ylim__)
        self._resetFrame()
        self.fig.canvas.draw()

    def _releaseHandler(self, evt):
        if evt.button != 1: return
        if evt.dblclick: return
        self._resetFrame()
        self.fig.canvas.draw()

    #### common plot type for maps: follow mpl convention ####
    def plot(self, ra, dec, *args, **kwargs):
        x, y = self.proj.transform(ra, dec)
        return self.ax.plot(x, y, *args, **kwargs)

    def scatter(self, ra, dec, **kwargs):
        x, y = self.proj.transform(ra, dec)
        return self.ax.scatter(x, y, **kwargs)

    def hexbin(self, ra, dec, C=None, **kwargs):
        x, y = self.proj.transform(ra, dec)
        # determine proper gridsize: by default x is only needed, y is chosen accordingly
        gridsize = kwargs.pop("gridsize", None)
        mincnt = kwargs.pop("mincnt", 1)
        if gridsize is None:
            xlim, ylim = (x.min(), x.max()), (y.min(), y.max())
            per_sample_volume = (xlim[1]-xlim[0])**2 / x.size * 10
            gridsize = int(np.ceil((xlim[1]-xlim[0]) / np.sqrt(per_sample_volume)))
        return self.ax.hexbin(x, y, C=C, gridsize=gridsize, mincnt=mincnt, **kwargs)

    def text(self, ra, dec, s, rotation=None, direction="parallel", **kwargs):
        x, y = self.proj.transform(ra, dec)

        if rotation is None:
            dxy_ = self.gradient(ra, dec, direction=direction)
            angle = 90-np.arctan2(dxy_[0], dxy_[1]) / DEG2RAD
        else:
            angle = rotation

        return self.ax.text(x, y, s, rotation=angle, rotation_mode="anchor", clip_on=True, **kwargs)

    def show(self, *args, **kwargs):
        self.fig.show(*args, **kwargs)

    def savefig(self, *args, **kwargs):
        self.fig.savefig(*args, **kwargs)

    #### special plot types for maps ####
    def footprint(self, surveyname, **kwargs):
        """Plot survey footprint polygon onto map

        Args:
            surveyname: name of the survey
            **kwargs: matplotlib.collections.PolyCollection keywords
        """
        ra, dec = footprint_loader[surveyname]()
        x,y  = self.proj.transform(ra, dec)
        from matplotlib.patches import Polygon
        poly = Polygon(np.dstack((x,y))[0], closed=True, **kwargs)
        self.ax.add_artist(poly)
        return poly

    def vertex(self, vertices, color=None, vmin=None, vmax=None, **kwargs):
        """Plot polygons (e.g. Healpix vertices)

        Args:
            vertices: cell boundaries in RA/Dec, from getCountAtLocations()
            color: string or matplib color, or numeric array to set polygon colors
            vmin: if color is numeric array, use vmin to set color of minimum
            vmax: if color is numeric array, use vmin to set color of minimum
            **kwargs: matplotlib.collections.PolyCollection keywords
        Returns:
            matplotlib.collections.PolyCollection
        """
        from matplotlib.collections import PolyCollection
        vertices_ = np.empty_like(vertices)
        vertices_[:,:,0], vertices_[:,:,1] = self.proj.transform(vertices[:,:,0], vertices[:,:,1])
        coll = PolyCollection(vertices_, array=color, **kwargs)
        coll.set_clim(vmin=vmin, vmax=vmax)
        coll.set_edgecolor("face")
        self.ax.add_collection(coll)
        return coll

    def healpix(self, m, nest=False, color_percentiles=[10,90], **kwargs):
        """Plot HealPix map

        Args:
            m: Healpix map array
            nest: HealPix nest
            color_percentiles: lower and higher cutoff percentile for map coloring
        """
        # determine ra, dec of map; restrict to non-empty cells
        pixels = np.flatnonzero(m)
        nside = healpix.hp.npix2nside(m.size)
        vertices = healpix.getHealpixVertices(pixels, nside, nest=nest)
        color = m[pixels]

        # styling
        cmap = kwargs.pop("cmap", "YlOrRd")
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        if vmin is None or vmax is None:
            vlim = np.percentile(color, color_percentiles)
            if vmin is None:
                vmin = vlim[0]
            if vmax is None:
                vmax = vlim[1]

        # make a map of the vertices
        return self.vertex(vertices, color=color, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)

    def density(self, ra, dec, nside=1024, color_percentiles=[10,90], **kwargs):
        """Plot sample density using healpix binning

        Args:
            ra: list of rectascensions
            dec: list of declinations
            nside: HealPix nside
            color_percentiles: lower and higher cutoff percentile for map coloring
        """
        # get count in healpix cells, restrict to non-empty cells
        bc, _, _, vertices = healpix.getCountAtLocations(ra, dec, nside=nside, return_vertices=True)
        color = bc

        # styling
        cmap = kwargs.pop("cmap", "YlOrRd")
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        if vmin is None or vmax is None:
            vlim = np.percentile(color, color_percentiles)
            if vmin is None:
                vmin = vlim[0]
            if vmax is None:
                vmax = vlim[1]

        # make a map of the vertices
        return self.vertex(vertices, color=color, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)

    def interpolate(self, ra, dec, value, **kwargs):
        """Interpolate ra,dec samples over covered region in the map

        Requires scipy, uses `scipy.interpolate.griddata` with `method='cubic'`.

        Args:
            ra: list of rectascensions
            dec: list of declinations
            value: list of sample values
            **kwargs: arguments for matplotlib.imshow
        """
        x, y = self.proj.transform(ra, dec)

        # evaluate interpolator over the range covered by data
        xlim, ylim = (x.min(), x.max()), (y.min(), y.max())
        per_sample_volume = min(xlim[1]-xlim[0], ylim[1]-ylim[0])**2 / x.size
        delta = np.sqrt(per_sample_volume)
        xline = np.arange(xlim[0]-delta/2, xlim[1]+delta/2, delta)
        yline = np.arange(ylim[0]-delta/2, ylim[1]+delta/2, delta)
        xp, yp = np.meshgrid(xline, yline)

        from scipy.interpolate import griddata
        vp = griddata(np.dstack((x,y))[0], value, (xp,yp), method='cubic')
        # remember axes limits ...
        xlim_, ylim_ = self.ax.get_xlim(), self.ax.get_ylim()
        _ = kwargs.pop('extend', None)
        artist = self.ax.imshow(vp, extent=(xlim[0], xlim[1], ylim[0], ylim[1]), **kwargs)
        # ... because imshow focusses on extent
        self.ax.set_xlim(xlim_)
        self.ax.set_ylim(ylim_)
        return artist

    def extrapolate(self, ra, dec, value, resolution=300, **kwargs):
        """Extrapolate ra,dec samples over entire map

        Requires scipy, uses default `scipy.interpolate.Rbf`.

        Args:
            ra: list of rectascensions
            dec: list of declinations
            value: list of sample values
            resolution: number of evaluated cells per linear map dimension
            **kwargs: arguments for matplotlib.imshow
        """

        x, y = self.proj.transform(ra, dec)
        # TODO: get limits of the map from all x/y data of the edge artists:
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        xline = np.linspace(xlim[0], xlim[1], resolution)
        yline = np.linspace(ylim[0], ylim[1], resolution)
        xp, yp = np.meshgrid(xline, yline)
        inside = self.proj.contains(xp,yp)
        vp = np.ma.array(np.empty(xp.shape), mask=~inside)
        from scipy.interpolate import Rbf
        rbfi = Rbf(x, y, value)
        vp[inside] = rbfi(xp[inside], yp[inside])
        return self.ax.imshow(vp, **kwargs, extent=(xlim[0], xlim[1], ylim[0], ylim[1]))


##### Start of free methods #####

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
