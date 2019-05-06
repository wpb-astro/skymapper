import healpy as hp
import numpy as np

# python 3 compatible
try:
    xrange
except NameError:
    xrange = range

def getHealpixArea(nside):
    return hp.nside2pixarea(nside, degrees=True)

def getHealpixVertices(pixels, nside, nest=False):
    """Get polygon vertices for list of HealPix pixels.

    Args:
        pixels: list of HealPix pixels
        nside: HealPix nside
        nest: HealPix nesting scheme

    Returns:
        vertices: (N,4,2), RA/Dec coordinates of 4 boundary points of cell
    """
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

def getGrid(nside, nest=False, return_vertices=False):
    pixels = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside, pixels, nest=nest)
    ra = phi*180/np.pi
    dec = 90 - theta*180/np.pi
    if return_vertices:
        vertices = getHealpixVertices(pixels, nside, nest=nest)
        return pixels, ra, dec, vertices
    return pixels, ra, dec

def getCountAtLocations(ra, dec, nside=512, per_area=True, return_vertices=False):
    """Get number density of objects from RA/Dec in HealPix cells.

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
    return bc, ra_, dec_

def reduceAtLocations(ra, dec, value, reduce_fct=np.mean, nside=512, return_vertices=False):
    """Reduce values at given RA/Dec in HealPix cells to a scalar.

    Args:
        ra: list of rectascensions
        dec: list of declinations
        value: list of values to be reduced
        reduce_fct: function to operate on values in each cell
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
    return v, ra_, dec_
