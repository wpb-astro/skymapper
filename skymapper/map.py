import matplotlib
import numpy as np
import re, pickle
from . import healpix
from . import survey_register
from matplotlib.patches import Polygon

DEG2RAD = np.pi/180

def skyDistance(radec, radec_ref):
    ra, dec = radec
    ra_ref, dec_ref = radec_ref
    ra_diff = np.abs(ra - ra_ref)
    mask = ra_diff > 180
    ra_diff[mask] = 360 - ra_diff[mask]
    ra_diff *= np.cos(dec_ref*DEG2RAD)
    dec_diff = dec - dec_ref
    return np.sqrt(ra_diff**2 + dec_diff**2)

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

def degPMFormatter(deg):
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

def deg360Formatter(deg):
        """Default formatter for map labels.

        Args:
            deg: float
        Returns:
            string
        """
        if deg < 0:
            deg += 360
        return "$%d^\circ$" % deg

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

def _parseArgs(locals):
    # turn list of arguments (all named or kwargs) into lfat dictionary
    locals.pop('self')
    kwargs = locals.pop('kwargs', {})
    for k,v in kwargs.items():
        locals[k] = v
    return locals

class Map():
    def __init__(self, proj, ax=None, resolution=75, interactive=True, **kwargs):
        # store arguments to regenerate the map
        self._config = {'__init__': _parseArgs(locals())}
        self.proj = proj
        self.resolution = resolution
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
        # do not unset the x/y ticks by e.g. xticks([]), we need them for tight_layout
        self.ax.xaxis.set_ticks_position('none')
        self.ax.yaxis.set_ticks_position('none')
        self.fig.tight_layout()

        # attach event handlers
        if interactive:
            self.fig.show()
            self._press_evt = self.fig.canvas.mpl_connect('button_press_event', self._pressHandler)
            self._release_evt = self.fig.canvas.mpl_connect('button_release_event', self._releaseHandler)
            self._scroll_evt = self.fig.canvas.mpl_connect('scroll_event', self._scrollHandler)

    def clone(self, ax=None):
        config = dict(self._config)
        config['xlim'] = self.ax.get_xlim()
        config['ylim'] = self.ax.get_ylim()
        return Map._create(config)

    def save(self, filename):
        try:
            with open(filename, 'wb') as fp:
                config = dict(self._config)
                config['xlim'] = self.ax.get_xlim()
                config['ylim'] = self.ax.get_ylim()
                pickle.dump(config, fp)
        except IOError as e:
            raise

    @staticmethod
    def load(filename, ax=None):
        try:
            with open(filename, 'rb') as fp:
                config = pickle.load(fp)
                fp.close()
                return Map._create(config)
        except IOError as e:
            raise

    @staticmethod
    def _create(config, ax=None):
        init_args = config.pop('__init__')
        init_args['ax'] = ax
        map = Map(**init_args)

        xlim = config.pop('xlim')
        ylim = config.pop('ylim')
        map.ax.set_xlim(xlim)
        map.ax.set_ylim(ylim)
        map._setFrame()

        meridian_args = config.pop('labelMeridianAtParallel')
        parallel_args = config.pop('labelParallelAtMeridian')
        for method in config.keys():
            getattr(map, method)(**config[method])

        for args in meridian_args:
            map.labelMeridianAtParallel(**args)

        for args in parallel_args:
            map.labelParallelAtMeridian(**args)

        map.fig.tight_layout()
        return map

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

    def _getParallel(self, p, reverse=False):
        if not reverse:
            return self.proj.transform(self._ra_range, p*np.ones(len(self._ra_range)))
        return self.proj.transform(self._ra_range[::-1], p*np.ones(len(self._ra_range)))

    def _getMeridian(self, m, reverse=False):
        if not reverse:
            return self.proj.transform(m*np.ones(len(self._dec_range)), self._dec_range)
        return self.proj.transform(m*np.ones(len(self._dec_range)), self._dec_range[::-1])

    def _setParallel(self, p, **kwargs):
        x_, y_ = self._getParallel(p)
        self.ax.plot(x_, y_, **kwargs)

    def _setMeridian(self, m, **kwargs):
        x_, y_ = self._getMeridian(m)
        self.ax.plot(x_, y_, **kwargs)

    def _setEdge(self, **kwargs):
        self._dec_range = np.linspace(-90, 90, self.resolution)
        self._ra_range = np.linspace(-180, 180, self.resolution) + self.proj.ra_0

        # styling: frame needs to be on top of everything, must be transparent
        facecolor = 'None'
        zorder = 1000
        lw = kwargs.pop('lw', 1)
        edgecolor = kwargs.pop('c', '#444444')
        # if there is facecolor: clone the polygon and put it in as bottom layer
        facecolor_ = kwargs.pop('facecolor', None)

        # polygon of the map edge: top, left, bottom, right
        # don't poles if that's a single point
        lines = []
        for line in [self._getParallel(90), self._getMeridian(self.proj.ra_0 + 180, reverse=True), self._getParallel(-90, reverse=True), self._getMeridian(self.proj.ra_0 - 180)]:
            if np.unique(line[0]).size > 1 and np.unique(line[1]).size > 1:
                lines.append(line)

        xy = np.concatenate(lines, axis=1).T
        self._edge = Polygon(xy, closed=True, edgecolor=edgecolor, facecolor=facecolor, lw=lw, zorder=zorder,gid="edge", **kwargs)
        self.ax.add_artist(self._edge)

        if facecolor_ is not None:
            zorder = -1000
            edgecolor = 'None'
            poly = Polygon(xy, closed=True, edgecolor=edgecolor, facecolor=facecolor_, zorder=zorder, gid="edge-background")
            self.ax.add_artist(poly)

    def xlim(self):
        return (self._edge.xy[:, 0].min(), self._edge.xy[:, 0].max())

    def ylim(self):
        return (self._edge.xy[:, 1].min(), self._edge.xy[:, 1].max())

    def grid(self, sep=30, parallel_fmt=degPMFormatter, meridian_fmt=deg360Formatter, dec_min=-90, dec_max=90, ra_min=-180, ra_max=180, **kwargs):
        self._config['grid'] = _parseArgs(locals())
        self.parallel_fmt = parallel_fmt
        self.meridian_fmt = meridian_fmt
        self._dec_range = np.linspace(dec_min, dec_max, self.resolution)
        self._ra_range = np.linspace(ra_min, ra_max, self.resolution) + self.proj.ra_0
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

        # styling
        ls = kwargs.pop('ls', '-')
        lw = kwargs.pop('lw', 0.5)
        c = kwargs.pop('c', 'k')
        alpha = kwargs.pop('alpha', 0.2)
        zorder = kwargs.pop('zorder', 10)

        for p in _parallels:
            self._setParallel(p, gid='grid-parallel-%r' % p, lw=lw, c=c, alpha=alpha, zorder=zorder, **kwargs)
        for m in _meridians:
            self._setMeridian(m, gid='grid-meridian-%r' % m, lw=lw, c=c, alpha=alpha, zorder=zorder, **kwargs)

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
        arguments = _parseArgs(locals())
        myname = 'labelMeridianAtParallel'
        if myname not in self._config.keys():
            self._config[myname] = []
        self._config[myname].append(arguments)

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
            dxy = self.proj.gradient(m, p, direction=direction)
            angle = np.arctan2(dxy[0], dxy[1]) / DEG2RAD
            options = np.arange(-2,3) * 90 # multiples of 90 deg
            closest = np.argmin(np.abs(options - angle))
            rot_base = options[closest]

        if meridians is None:
            meridians = self.meridians

        for m in meridians:
            # move label along meridian
            xp, yp = self.proj.transform(m, p)
            dxy = self.proj.gradient(m, p, direction="meridian")
            dxy *= pad / np.sqrt((dxy**2).sum())
            if loc == 'bottom':
                dxy *= -1

            if rotation is None:
                dxy_ = self.proj.gradient(m, p, direction=direction)
                angle = rot_base-np.arctan2(dxy_[0], dxy_[1]) / DEG2RAD
            else:
                angle = rotation

            self.ax.annotate(self.meridian_fmt(m), (xp, yp), xytext=dxy, textcoords='offset points', rotation=angle, rotation_mode='anchor', horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, size=size, zorder=zorder, gid='meridian-label', **kwargs)

    def labelParallelAtMeridian(self, m, loc=None, parallels=None, pad=None, direction='parallel', **kwargs):
        arguments = _parseArgs(locals())
        myname = 'labelParallelAtMeridian'
        if myname not in self._config.keys():
            self._config[myname] = []
        self._config[myname].append(arguments)

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
            dxy = self.proj.gradient(m, p, direction=direction)
            angle = np.arctan2(dxy[0], dxy[1]) / DEG2RAD
            options = np.arange(-2,3) * 90
            closest = np.argmin(np.abs(options - angle))
            rot_base = options[closest]

        if parallels is None:
            parallels = self.parallels

        for p in parallels:
            # move label along parallel
            xp, yp = self.proj.transform(m, p)
            dxy = self.proj.gradient(m, p, direction="parallel")
            dxy *= pad / np.sqrt((dxy**2).sum())
            if loc == 'left':
                dxy *= -1

            if rotation is None:
                dxy_ = self.proj.gradient(m, p, direction=direction)
                angle = rot_base-np.arctan2(dxy_[0], dxy_[1]) / DEG2RAD
            else:
                angle = rotation

            self.ax.annotate(self.parallel_fmt(p), (xp, yp), xytext=dxy, textcoords='offset points', rotation=angle, rotation_mode='anchor',  horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, size=size, zorder=zorder,  gid='parallel-label', **kwargs)

    def labelMeridiansAtFrame(self, loc='top', meridians=None, pad=None, **kwargs):
        assert loc in ['bottom', 'top']

        arguments = _parseArgs(locals())
        myname = 'labelMeridiansAtFrame'
        # need extra space for tight_layout to consider the frame annnotations
        # we can't get the actual width, but we can make use the of the default width of the axes tick labels
        if myname not in self._config.keys() or self._config[myname]['loc'] != loc:
            self.ax.xaxis.set_ticks_position(loc)
            self.ax.xaxis.set_label_position(loc)
            # remove existing
            frame_artists = self.artists('frame-meridian-label')
            for artist in frame_artists:
                artist.remove()
            self.fig.tight_layout()
        self._config[myname] = arguments

        size = kwargs.pop('size', matplotlib.rcParams['font.size'])
        zorder = kwargs.pop('zorder', self.ax.spines[loc].get_zorder())
        horizontalalignment = kwargs.pop('horizontalalignment', 'center')
        verticalalignment = self._negateLoc(loc) # no option along the frame
        _ = kwargs.pop('verticalalignment', None)

        if pad is None:
            pad = size / 3

        if meridians is None:
            meridians = self.meridians

        # check if loc has frame
        poss = {"bottom": 0, "top": 1}
        pos = poss[loc]
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        frame_artists = self.artists(r'frame-([a-zA-Z]+)', regex=True)
        frame_locs = [match.group(1) for c,match in frame_artists]
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
                        dxy = self.proj.gradient(m_, p_, direction="meridian")
                        dxy /= np.sqrt((dxy**2).sum())
                        dxy *= pad / dxy[1] # same pad from frame
                        if loc == "bottom":
                            dxy *= -1
                        angle = 0 # no option along the frame

                        x_im = (xm_at_ylim - xlim[0])/(xlim[1]-xlim[0])
                        y_im = (ylim[pos] - ylim[0])/(ylim[1]-ylim[0])

                        # these are set as annotations instead of simple axis ticks
                        # because those cannot be shifted by a constant point amount to
                        # follow the graticule
                        self.ax.annotate(self.meridian_fmt(m), (x_im, y_im), xycoords='axes fraction', xytext=dxy, textcoords='offset points', annotation_clip=False,  gid='frame-meridian-label', horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, size=size, zorder=zorder,  **kwargs)

    def labelParallelsAtFrame(self, loc='left', parallels=None, pad=None, **kwargs):
        assert loc in ['left', 'right']

        arguments = _parseArgs(locals())
        myname = 'labelParallelsAtFrame'
        # need extra space for tight_layout to consider the frame annnotations
        # we can't get the actual width, but we can make use the of the default width of the axes tick labels
        if myname not in self._config.keys() or self._config[myname]['loc'] != loc:
            self.ax.yaxis.set_ticks_position(loc)
            self.ax.yaxis.set_label_position(loc)
            # remove existing
            frame_artists = self.artists('frame-parallel-label')
            for artist in frame_artists:
                artist.remove()
            self.fig.tight_layout()
        self._config[myname] = arguments

        size = kwargs.pop('size', matplotlib.rcParams['font.size'])
        zorder = kwargs.pop('zorder', self.ax.spines[loc].get_zorder())
        verticalalignment = kwargs.pop('verticalalignment', 'center')
        horizontalalignment = self._negateLoc(loc) # no option along the frame
        _ = kwargs.pop('horizontalalignment', None)

        if pad is None:
            pad = size / 3

        if parallels is None:
            parallels = self.parallels

        # check if loc has frame
        poss = {"left": 0, "right": 1}
        pos = poss[loc]
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        frame_artists = self.artists(r'frame-([a-zA-Z]+)', regex=True)
        frame_locs = [match.group(1) for c,match in frame_artists]
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
                        dxy = self.proj.gradient(m_, p_, direction='parallel')
                        dxy /= np.sqrt((dxy**2).sum())
                        dxy *= pad / dxy[0] # same pad from frame
                        if loc == "left":
                            dxy *= -1
                        angle = 0 # no option along the frame

                        x_im = (xlim[pos] - xlim[0])/(xlim[1]-xlim[0])
                        y_im = (yp_at_xlim - ylim[0])/(ylim[1]-ylim[0])
                        # these are set as annotations instead of simple axis ticks
                        # because those cannot be shifted by a constant point amount to
                        # follow the graticule
                        self.ax.annotate(self.parallel_fmt(p), (x_im, y_im), xycoords='axes fraction', xytext=dxy, textcoords='offset points', annotation_clip=False, gid='frame-parallel-label', horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, size=size, zorder=zorder,  **kwargs)

    def _setFrame(self):
        # clean up existing frame
        frame_artists = self.artists(r'frame-([a-zA-Z]+)', regex=True)
        for c,m in frame_artists:
            c.remove()

        locs = ['left', 'bottom', 'right', 'top']
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()

        for loc in locs:
            # define line along axis
            precision = 1000
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
        self._setFrame()
        for method in ['labelMeridiansAtFrame', 'labelParallelsAtFrame']:
            getattr(self, method)(**self._config[method])

    def _pressHandler(self, evt):
        if evt.button != 1: return
        if evt.dblclick: return
        # remove frame and labels
        self._clearFrame()
        self.fig.canvas.draw()

    def _releaseHandler(self, evt):
        if evt.button != 1: return
        if evt.dblclick: return
        self._resetFrame()
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

        # styling: use same default colormap as density for histogram
        if C is None:
            cmap = kwargs.pop("cmap", "YlOrRd")
        else:
            cmap = kwargs.pop("cmap", "None")
        zorder = kwargs.pop("zorder", 0) # same as for imshow: underneath everything

        artist = self.ax.hexbin(x, y, C=C, gridsize=gridsize, mincnt=mincnt, cmap=cmap, zorder=zorder, **kwargs)
        artist.set_clip_path(self._edge)
        return artist

    def text(self, ra, dec, s, rotation=None, direction="parallel", **kwargs):
        x, y = self.proj.transform(ra, dec)

        if rotation is None:
            dxy_ = self.proj.gradient(ra, dec, direction=direction)
            angle = 90-np.arctan2(dxy_[0], dxy_[1]) / DEG2RAD
        else:
            angle = rotation

        return self.ax.text(x, y, s, rotation=angle, rotation_mode="anchor", clip_on=True, **kwargs)

    def colorbar(self, cb_collection, cb_label="", orientation="vertical", size="2%", pad="1%"):
        assert orientation in ["vertical", "horizontal"]

        # pick the side that does not have the tick labels
        if orientation == "vertical":
            frame_loc = self._config['labelParallelsAtFrame']['loc']
        else:
            frame_loc = self._config['labelMeridiansAtFrame']['loc']
        loc = self._negateLoc(frame_loc)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes(loc, size=size, pad=pad)
        cb = self.fig.colorbar(cb_collection, cax=cax, orientation=orientation, ticklocation=loc)
        cb.solids.set_edgecolor("face")
        cb.set_label(cb_label)
        self.fig.tight_layout()
        return cb

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
        # search for survey in register
        ra, dec = survey_register[surveyname].get_footprint()

        x,y  = self.proj.transform(ra, dec)
        poly = Polygon(np.dstack((x,y))[0], closed=True, **kwargs)
        poly.set_clip_path(self._edge)
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
        vertices_ = np.empty_like(vertices)
        vertices_[:,:,0], vertices_[:,:,1] = self.proj.transform(vertices[:,:,0], vertices[:,:,1])

        from matplotlib.collections import PolyCollection
        zorder = kwargs.pop("zorder", 0) # same as for imshow: underneath everything
        coll = PolyCollection(vertices_, array=color, zorder=zorder, **kwargs)
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
        zorder = kwargs.pop("zorder", 0) # same as for imshow: underneath everything
        if vmin is None or vmax is None:
            vlim = np.percentile(color, color_percentiles)
            if vmin is None:
                vmin = vlim[0]
            if vmax is None:
                vmax = vlim[1]

        # make a map of the vertices
        return self.vertex(vertices, color=color, vmin=vmin, vmax=vmax, cmap=cmap, zorder=zorder, **kwargs)

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
        zorder = kwargs.pop("zorder", 0) # same as for imshow: underneath everything
        if vmin is None or vmax is None:
            vlim = np.percentile(color, color_percentiles)
            if vmin is None:
                vmin = vlim[0]
            if vmax is None:
                vmax = vlim[1]

        # make a map of the vertices
        return self.vertex(vertices, color=color, vmin=vmin, vmax=vmax, cmap=cmap, zorder=zorder, **kwargs)

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
        zorder = kwargs.pop("zorder", 0) # default for imshow: underneath everything
        artist = self.ax.imshow(vp, extent=(xlim[0], xlim[1], ylim[0], ylim[1]), zorder=zorder, **kwargs)
        artist.set_clip_path(self._edge)
        # ... because imshow focusses on extent
        self.ax.set_xlim(xlim_)
        self.ax.set_ylim(ylim_)
        return artist

    def extrapolate(self, ra, dec, value, resolution=300, **kwargs):
        """Extrapolate ra,dec samples on the entire sphere and project on the map

        Requires scipy, uses default `scipy.interpolate.Rbf`.

        Args:
            ra: list of rectascensions
            dec: list of declinations
            value: list of sample values
            resolution: number of evaluated cells per linear map dimension
            **kwargs: arguments for matplotlib.imshow
        """
        # interpolate samples in RA/DEC
        from scipy.interpolate import Rbf
        rbfi = Rbf(ra, dec, value, norm=skyDistance)

        # make grid in x/y over the limits of the map
        xlim, ylim = self.xlim(), self.ylim()
        if resolution % 1 == 0:
            resolution += 1
        dx = (xlim[1]-xlim[0])/resolution
        xline = np.arange(xlim[0], xlim[1], dx) + dx/2 # evaluate center pixel
        yline = np.arange(ylim[0], ylim[1], dx) + dx/2
        xp, yp = np.meshgrid(xline, yline)
        inside = self.proj.contains(xp,yp)
        vp = np.ma.array(np.empty(xp.shape), mask=~inside)

        rap, decp = self.proj.invert(xp[inside], yp[inside])
        vp[inside] = rbfi(rap, decp)
        zorder = kwargs.pop("zorder", 0) # same as for imshow: underneath everything
        xlim_, ylim_ = self.ax.get_xlim(), self.ax.get_ylim()
        artist = self.ax.imshow(vp, extent=(xline[0], xline[-1], yline[0], yline[-1]), zorder=zorder, **kwargs)
        artist.set_clip_path(self._edge)
        # ... because imshow focusses on extent
        self.ax.set_xlim(xlim_)
        self.ax.set_ylim(ylim_)
        return artist
