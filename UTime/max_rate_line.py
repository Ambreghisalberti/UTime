"""Line of maximum of a 2D map (e.g. reconnection rate) on the magnetopause.

Verbatim copy of ``shear_maps.make_max_qty_from_max_point`` and of the helpers it uses
(``shear_maps`` and ``utilities`` from /DATA/michotte/xline), with ONE change: the call to
``skimage.feature.hessian_matrix`` (see ``hessian_matrix`` below).

Why: in scikit-image <= 0.19, ``hessian_matrix(qty)`` (default ``order='rc'``) actually returned
the elements in reversed axis order, i.e. ``[Hcc, Hrc, Hrr]``, and ``shear_maps`` relies on that.
The bug was fixed in scikit-image 0.20, so with recent versions the y and z components of the
Hessian eigenvectors are swapped and the traced "max line" goes in the wrong direction.
In scikit-image >= 0.20, ``order='xy'`` gives exactly what ``order='rc'`` gave in 0.19.
"""
import numpy as np
import skimage
from scipy.integrate import solve_ivp
from scipy.interpolate import LinearNDInterpolator
from skimage.feature import peak_local_max
from skimage.feature import hessian_matrix as _skimage_hessian_matrix
from spok import smath as sm
from spok import utils as su


def hessian_matrix(qty):
    """Same output as ``skimage.feature.hessian_matrix(qty)`` with scikit-image 0.19, whatever the installed version."""
    major, minor = (int(v) for v in skimage.__version__.split('.')[:2])
    if (major, minor) < (0, 20):
        return _skimage_hessian_matrix(qty)
    return _skimage_hessian_matrix(qty, order='xy', use_gaussian_derivatives=False)


# ---------------------------------------------------------------------------------------------
# Unchanged copies from utilities.py
# ---------------------------------------------------------------------------------------------

def reshape_to_2Darrays(lst_arrays):
    lst_arrays = [np.array(su.listify(el)) for el in lst_arrays]
    if np.sum([lst_arrays[0].shape != el.shape for el in lst_arrays[1:]]):
        raise ValueError('All elements of lst_arrays must have the same shape')
    if len(lst_arrays[0].shape) == 1:
        a2d = np.array(lst_arrays).T
        old_shape = np.array(lst_arrays).shape
    else:
        a2d = np.asarray(lst_arrays)
        old_shape = a2d.shape
        a2d = a2d.T.ravel().reshape(np.prod(old_shape[1:]), old_shape[0])
    return a2d, old_shape


# ---------------------------------------------------------------------------------------------
# Unchanged copies from shear_maps.py
# ---------------------------------------------------------------------------------------------

def make_hessian_e2_vector(x, y, qty):
    Hrr, Hrc, Hcc = hessian_matrix(qty)
    mat = np.zeros((len(x), len(y), 2, 2))
    mat[:, :, 0, 0] = Hrr
    mat[:, :, 1, 0] = Hrc
    mat[:, :, 0, 1] = Hrc
    mat[:, :, 1, 1] = Hcc
    hess_val, hess_vec = np.linalg.eigh(mat)
    return hess_vec[:, :, :, 1]


def make_linear_interpolator(x, y, qty):
    arr2d = reshape_to_2Darrays([x, y, qty])[0]
    return LinearNDInterpolator(arr2d[:, :2], arr2d[:, -1])


def make_hessian_e2_interpolator(x, y, qty, indexing='xy'):
    if indexing == 'xy':
        yaxis = 0
        zaxis = 1
    else:
        yaxis = 1
        zaxis = 0
    e2 = make_hessian_e2_vector(x, y, qty)
    e2x = make_linear_interpolator(x, y, e2[:, :, yaxis])
    e2y = make_linear_interpolator(x, y, e2[:, :, zaxis])
    return e2x, e2y


def find_max_points(xx, yy, qty, rlim_max=9, min_distance=1):
    coord = peak_local_max(qty, min_distance=min_distance)
    xm = np.asarray([xx[c[0], c[1]] for c in coord])
    ym = np.asarray([yy[c[0], c[1]] for c in coord])
    qm = np.asarray([qty[c[0], c[1]] for c in coord])
    r = sm.norm(xm, ym, 0)
    xm, ym, qm = xm[r <= rlim_max], ym[r <= rlim_max], qm[r <= rlim_max]
    return xm, ym, qm


def outofbounds(t, pos, interxy, rlim):
    if np.sqrt(pos[0] ** 2 + pos[1] ** 2) > rlim:
        v = 0
    else:
        v = 1
    return v


outofbounds.terminal = True


def get_line_with_hess2(interhess2,
                        x0=0,
                        y0=0,
                        t0=0,
                        tfinal=100,
                        fac=1,
                        max_step=0.05, first_step=0.05, rlim=15,
                        outofbounds=outofbounds):
    def vel(t,  # pseudo time
            pos,  # x and y positions
            interhess2, rlim):  # eigenvector interpolators in x and y directions:   # some arbitrary magnification coef
        vv = [fac * (interhess2[0](pos[0], pos[1])),
              fac * (interhess2[1](pos[0], pos[1]))]

        return vv

    return solve_ivp(vel, [t0, tfinal], [x0, y0],
                     args=(interhess2, rlim),
                     method="BDF", events=outofbounds, max_step=max_step, first_step=max_step).y


def find_lines_following_hessian_from_max_point(xm, ym, ie2, rlim=16):
    part1 = get_line_with_hess2(ie2, x0=xm, y0=ym, fac=-1, rlim=rlim)
    part2 = get_line_with_hess2(ie2, x0=xm, y0=ym, fac=1, rlim=rlim)
    if sm.norm(0, part1[0][-1], part1[1][-1]) > sm.norm(0, part2[0][-1], part2[1][-1]):
        line = np.concatenate([part2[0][::-1], part1[0]]), np.concatenate([part2[1][::-1], part1[1]])
    elif sm.norm(0, part1[0][-1], part1[1][-1]) < sm.norm(0, part2[0][-1], part2[1][-1]):
        line = np.concatenate([part1[0][::-1], part2[0]]), np.concatenate([part1[1][::-1], part2[1]])
    return line


def make_max_qty_from_max_point(xx, yy, qty, rlim=16, rlim_subsolar=None, indexing='xy'):
    if rlim_subsolar:
        xm, ym, qm = find_max_points(xx, yy, qty, rlim_max=rlim_subsolar)
    else:
        xm, ym, qm = find_max_points(xx, yy, qty, rlim_max=rlim)
    xm, ym = xm[np.argmax(qm)], ym[np.argmax(qm)]
    ie2 = make_hessian_e2_interpolator(xx, yy, qty, indexing=indexing)
    line = find_lines_following_hessian_from_max_point(xm, ym, ie2, rlim=rlim)
    return line
