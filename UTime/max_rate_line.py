"""Line of maximum of a 2D map (e.g. reconnection rate) on the magnetopause.

Self-contained, version-independent copy of ``shear_maps.make_max_qty_from_max_point``
(and the few helpers it needs).

Why this exists: ``shear_maps`` builds the Hessian with ``skimage.feature.hessian_matrix``.
In scikit-image <= 0.19, ``order='rc'`` (the default) actually returned the elements in
reversed axis order, i.e. ``[Hcc, Hrc, Hrr]``, and ``shear_maps`` relies on that. The bug was
fixed in scikit-image 0.20, so with recent versions the y and z components of the Hessian
eigenvectors are swapped and the traced "max line" goes in the wrong direction.
Here the Hessian is computed explicitly (Gaussian smoothing + finite differences, exactly as
scikit-image 0.19 did), so the result no longer depends on the installed scikit-image version.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max


def norm(u, v, w):
    return np.sqrt(u ** 2 + v ** 2 + w ** 2)


def _axes(indexing='xy'):
    """Array axes along which the first (y) and second (z) coordinates vary."""
    if indexing == 'xy':
        return 1, 0
    return 0, 1


def hessian_yz(qty, sigma=1, indexing='xy'):
    """Hessian elements (Hyy, Hyz, Hzz) of qty, in grid-point units.

    Same computation as scikit-image 0.19 ``hessian_matrix`` (Gaussian filter with
    mode='constant', cval=0, then two successive ``np.gradient``), but with the axes
    ordering made explicit.
    """
    yaxis, zaxis = _axes(indexing)
    smoothed = gaussian_filter(np.asarray(qty, dtype=float), sigma=sigma, mode='constant', cval=0)
    grad = np.gradient(smoothed)
    hyy = np.gradient(grad[yaxis], axis=yaxis)
    hyz = np.gradient(grad[yaxis], axis=zaxis)
    hzz = np.gradient(grad[zaxis], axis=zaxis)
    return hyy, hyz, hzz


def make_hessian_e2_vector(qty, indexing='xy'):
    """Eigenvector of the Hessian with the largest eigenvalue, as (e2y, e2z).

    Along a ridge (line of maximum) this is the direction of the ridge.
    """
    hyy, hyz, hzz = hessian_yz(qty, indexing=indexing)
    mat = np.zeros(qty.shape + (2, 2))
    mat[..., 0, 0] = hyy
    mat[..., 0, 1] = hyz
    mat[..., 1, 0] = hyz
    mat[..., 1, 1] = hzz
    hess_val, hess_vec = np.linalg.eigh(mat)
    e2 = hess_vec[..., :, 1]
    return e2[..., 0], e2[..., 1]


def make_linear_interpolator(x, y, qty):
    return LinearNDInterpolator(np.array([np.ravel(x), np.ravel(y)]).T, np.ravel(qty))


def make_hessian_e2_interpolator(x, y, qty, indexing='xy'):
    e2y, e2z = make_hessian_e2_vector(qty, indexing=indexing)
    return make_linear_interpolator(x, y, e2y), make_linear_interpolator(x, y, e2z)


def find_max_points(x, y, qty, rlim_max=9, min_distance=1):
    coord = peak_local_max(qty, min_distance=min_distance)
    xm = np.asarray([x[c[0], c[1]] for c in coord])
    ym = np.asarray([y[c[0], c[1]] for c in coord])
    qm = np.asarray([qty[c[0], c[1]] for c in coord])
    r = norm(xm, ym, 0)
    return xm[r <= rlim_max], ym[r <= rlim_max], qm[r <= rlim_max]


def outofbounds(t, pos, interxy, rlim):
    if np.sqrt(pos[0] ** 2 + pos[1] ** 2) > rlim:
        return 0
    return 1


outofbounds.terminal = True


def get_line_with_hess2(interhess2, x0=0, y0=0, t0=0, tfinal=100, fac=1, max_step=0.05, rlim=15):
    def vel(t, pos, interhess2, rlim):
        return [fac * float(np.squeeze(interhess2[0](pos[0], pos[1]))),
                fac * float(np.squeeze(interhess2[1](pos[0], pos[1])))]

    return solve_ivp(vel, [t0, tfinal], [x0, y0], args=(interhess2, rlim), method="BDF",
                     events=outofbounds, max_step=max_step, first_step=max_step).y


def find_lines_following_hessian_from_max_point(xm, ym, ie2, rlim=16):
    part1 = get_line_with_hess2(ie2, x0=xm, y0=ym, fac=-1, rlim=rlim)
    part2 = get_line_with_hess2(ie2, x0=xm, y0=ym, fac=1, rlim=rlim)
    if norm(0, part1[0][-1], part1[1][-1]) >= norm(0, part2[0][-1], part2[1][-1]):
        return np.concatenate([part2[0][::-1], part1[0]]), np.concatenate([part2[1][::-1], part1[1]])
    return np.concatenate([part1[0][::-1], part2[0]]), np.concatenate([part1[1][::-1], part2[1]])


def make_max_qty_from_max_point(xx, yy, qty, rlim=16, rlim_subsolar=None, indexing='xy'):
    """Line of maximum of qty, integrated along the Hessian ridge direction from the global max point.

    xx, yy : 2D regular grid (as given by np.meshgrid with the given indexing)
    qty : 2D map on that grid
    Returns (y, z) arrays of the line positions.
    """
    xm, ym, qm = find_max_points(xx, yy, qty, rlim_max=rlim_subsolar if rlim_subsolar else rlim)
    xm, ym = xm[np.argmax(qm)], ym[np.argmax(qm)]
    ie2 = make_hessian_e2_interpolator(xx, yy, qty, indexing=indexing)
    return find_lines_following_hessian_from_max_point(xm, ym, ie2, rlim=rlim)
