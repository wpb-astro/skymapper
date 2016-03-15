# Skymapper

A number of python / matplotlib scripts to map astronomical survey data from the celestial sphere onto 2D. The purpose of these methods is to facilitate the creating of publication-quality plots (as well as interactive graphics or movies) with a python-based workflow many astronomers are accustomed with.

The code requires matplotlib and numpy, but is independent of Basemap, which is not part of matplotlib distributions any more and can be *very* troublesome to install.

Currently the only map projection available is **Albers Equal Area conic** projection (an explanation why exactly that one is coming soon). This can be used to plot any point data or HealPix polygons onto regular matplotlib axes.

More projections and plot types will be added as needed. Open an issue for any such request.

##Limitation(s)

The projection is not a full-fledged [matplotlib transformation](http://matplotlib.org/users/transforms_tutorial.html). It does allow you to zoom and drag interactively, but the tick labels and the mouse position do not update correctly. The underlying reason is that the map pretends (x,y) to be independent variables, which isn't true for a non-linear transformation.

While perflectly sufficient for publication-quality plots, it's not ideal for exploratory work. A (failed) attempt of implementing a complete transform is [here](aea_projection.py), but it throws some gibberish error message. It's based on [this example](http://matplotlib.org/examples/api/custom_projection_example.html), so if you feel up to it: please give it a try!

## Example use

```python
# load projection and helper functions
from skymapper import *

# load RA/Dec from catalog
import fitsio
fits = fitsio.FITS(catalogfile)
w = fits[1].where('DEC < - 35')
ra_dec = fits[1]['RA', 'DEC'][w]
fits.close()

# get count in healpix cells
nside = 512
bc, ra, dec, vertices = getCountAtLocations(ra_dec['RA'], ra_dec['DEC'], nside=nside, return_vertices=True)

# setup figure
import matplotlib.cm as cm
cmap = cm.YlOrRd
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, aspect='equal')

# setup map: define AEA map optimal for given RA/Dec
aea = createAEAMap(ax, ra, dec)
# add lines and labels for meridians (separation 5 deg, south)
# and parallels (separation 1h = 15 deg)
meridians = np.linspace(-90, 0, 19)
parallels = np.linspace(0, 360, 25)
aea.setMeridianPatches(ax, meridians, linestyle=':', lw=0.5, zorder=1)
aea.setParallelPatches(ax, parallels, linestyle=':', lw=0.5, zorder=1)
aea.setMeridianLabels(ax, meridians, loc="left", fmt=pmDegFormatter)
aea.setParallelLabels(ax, parallels, loc="top", fmt=hourAngleFormatter)

# add healpix counts from vertices
vmin = 1
vmax = 2
poly = plotHealpixPolygons(ax, aea, vertices, color=bc, vmin=vmin, vmax=vmax, cmap=cmap, zorder=2, rasterized=True)

# add another data set (clusters with richness lambda)
fits = fitsio.FITS(catalogfile)
w = fits[1].where('LAMBDA_CHISQ > 50')
clusters = fits[1]['RA', 'DEC', 'LAMBDA_CHISQ'][w]
fits.close()
x,y = aea(clusters['RA'], clusters['DEC'])
scc = ax.scatter(x,y, c='None', s=clusters['LAMBDA_CHISQ']/4, edgecolors='#2B3856', linewidths=1, marker='o', zorder=3)

# add colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.0)
cb = plt.colorbar(poly, cax=cax)
cb.set_label('$n_{gal}$ [arcmin$^{-2}$]')
ticks = np.linspace(vmin, vmax, 5)
cb.set_ticks(ticks)
cb.solids.set_edgecolor("face")

# show (and save)
plt.show()
fig.savefig(filename)
```

