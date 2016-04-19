#!/bin/env python

import skymapper
import numpy as np
import matplotlib.pyplot as plt

def getCatalog():
    # dummy catalog 
    ra = np.random.uniform(size=10000, low=45, high=75)
    dec = np.random.uniform(size=10000, low=-54, high=-44)
    z = np.random.uniform(size=10000, low=0.2, high=0.6)
    return ra, dec, z


if __name__ == "__main__":

    # load RA, Dec, z ...
    ra, dec, z =  getCatalog()

    # set slice thickness and method label
    dz = 0.03 # roughly 100 Mpc/h along the LoS
    method = "MY_METHOD"
    reticule = 5. # separation of sky coordinate grid lines
    overlapping = True

    # iterate in overlapping z slices
    z0 = 0
    z_max = z.max()

    # need to fix one projection and map
    proj = None
    ax0 = None
    while z0 <= z_max:
        sel = (z >= z0) & (z < z0 + dz)
        z_mean = z0 + dz/2
        if sel.any():

            # create skymap:
            # store axes in ax0 to prevent adjustments in x/y limits
            fig = plt.figure(figsize=(10,5))
            ax = fig.add_subplot(111, aspect='equal')
            if proj is None:
                # define projection and reference ax to hold
                proj = skymapper.createConicMap(ax, ra, dec, bgcolor='w')
                ax0 = ax
            else:
                skymapper.cloneMap(ax0, ax)

            # plot data in slice
            x,y = proj(ra[sel], dec[sel])
            ax.scatter(x,y, s=2, marker='o', c='k', edgecolor='None', zorder=5)

            # add labels and ax
            parallels = np.arange(0. ,360., reticule)
            meridians = np.arange(-90., 90., reticule)
            skymapper.setMeridianPatches(ax, proj, meridians, linestyle='-', lw=0.5, alpha=0.3, zorder=3)
            skymapper.setParallelPatches(ax, proj, parallels, linestyle='-', lw=0.5, alpha=0.3, zorder=3)
            skymapper.setParallelLabels(ax, proj, parallels, loc="bottom")
            skymapper.setMeridianLabels(ax, proj, meridians, loc="left")
            ax.set_title('%s: $z = %.3f\pm%.3f$' % (method, z_mean, dz/2))

            fig.tight_layout()
            fig.show()
            fig.savefig('%s_z%.3f.png' % (method, z_mean), dpi=72)
        if overlapping:
            # ensure slices are overlapping by 50% to avoid cutting structures
            z0 += dz/2
        else:
            z0 += dz
