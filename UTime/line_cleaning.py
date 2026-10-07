"""Remove the bits of a line that are far away from its main part.

A line retrieved from a map (max rate line, max shear line, ...) is a set of points that can be
discontinuous. Some pieces can however be far from the rest and are then not physical.
Points are grouped into pieces: two points belong to the same piece if they can be linked by a
chain of points that are each less than ``max_gap`` apart. The main piece (the one with the
largest spatial extent) is kept; the other pieces are removed, unless they are large enough
(``min_fraction``).
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def label_pieces(y, z, max_gap=2.):
    """Label of the piece each point belongs to (-1 for NaN points)."""
    y, z = np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    labels = np.full(y.shape, -1)
    valid = np.isfinite(y) & np.isfinite(z)
    pts = np.array([y[valid], z[valid]]).T
    if len(pts) == 0:
        return labels
    pairs = cKDTree(pts).query_pairs(max_gap, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pts), len(pts)))
    labels[valid] = connected_components(graph, directed=False)[1]
    return labels


def piece_extent(y, z):
    """Size of a piece: diagonal of its bounding box."""
    return np.hypot(np.ptp(y), np.ptp(z))


def keep_main_part(y, z, max_gap=2., min_fraction=None, return_mask=False):
    """Remove the pieces of the line (y, z) that are far from its main part.

    y, z : positions of the points of the line (the order is kept)
    max_gap : points closer than max_gap are in the same piece. Should be larger than the
        gaps you accept in a line, and smaller than the distance of the pieces to remove.
    min_fraction : if given, also keep the pieces whose extent is at least min_fraction times
        the extent of the main piece.
    Returns the cleaned y, z (and the boolean mask of the kept points if return_mask).
    """
    y, z = np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    labels = label_pieces(y, z, max_gap=max_gap)
    pieces = np.unique(labels[labels >= 0])
    if len(pieces) == 0:
        mask = np.zeros(y.shape, dtype=bool)
    else:
        extents = np.array([piece_extent(y[labels == p], z[labels == p]) for p in pieces])
        if min_fraction is None:
            kept = pieces[[np.argmax(extents)]]
        else:
            kept = pieces[extents >= min_fraction * extents.max()]
        mask = np.isin(labels, kept)
    if return_mask:
        return y[mask], z[mask], mask
    return y[mask], z[mask]
