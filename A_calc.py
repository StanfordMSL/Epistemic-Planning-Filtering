# -----------------------------------------------------------------------------
# This file was autogenerated by symforce from template:
#     backends/python/templates/function/FUNCTION.py.jinja
# Do NOT modify by hand.
# -----------------------------------------------------------------------------

import math  # pylint: disable=unused-import
import numpy  # pylint: disable=unused-import
import typing as T  # pylint: disable=unused-import

import sym  # pylint: disable=unused-import


# pylint: disable=too-many-locals,too-many-lines,too-many-statements,unused-argument


def a_calc(x, u, dt, m):
    # type: (T.Sequence[float], T.Sequence[float], float, float) -> numpy.ndarray
    """
    This function was autogenerated. Do not modify by hand.

    Args:
        x: Matrix91
        u: Matrix41
        dt: Scalar
        m: Scalar

    Outputs:
        A: Matrix9_9
    """

    # Total ops: 70

    # Input arrays

    # Intermediate terms (22)
    _tmp0 = math.tan(x[4])
    _tmp1 = math.sin(x[3])
    _tmp2 = _tmp1 * u[3]
    _tmp3 = math.cos(x[3])
    _tmp4 = _tmp3 * u[2]
    _tmp5 = _tmp3 * u[3]
    _tmp6 = _tmp1 * u[2]
    _tmp7 = math.cos(x[4])
    _tmp8 = 1.0 / _tmp7
    _tmp9 = math.sin(x[5])
    _tmp10 = 1.0 * _tmp9
    _tmp11 = math.sin(x[4])
    _tmp12 = math.cos(x[5])
    _tmp13 = _tmp11 * _tmp12
    _tmp14 = dt * u[0] / m
    _tmp15 = 1.0 * _tmp12
    _tmp16 = _tmp11 * _tmp9
    _tmp17 = _tmp14 * _tmp7
    _tmp18 = _tmp0**2 + 1
    _tmp19 = 1.0 * _tmp11
    _tmp20 = _tmp19 / _tmp7**2
    _tmp21 = _tmp17 * _tmp3

    # Output terms
    _A = numpy.zeros((9, 9))
    _A[0, 0] = 1
    _A[1, 0] = 0
    _A[2, 0] = 0
    _A[3, 0] = 0
    _A[4, 0] = 0
    _A[5, 0] = 0
    _A[6, 0] = 0
    _A[7, 0] = 0
    _A[8, 0] = 0
    _A[0, 1] = 0
    _A[1, 1] = 1
    _A[2, 1] = 0
    _A[3, 1] = 0
    _A[4, 1] = 0
    _A[5, 1] = 0
    _A[6, 1] = 0
    _A[7, 1] = 0
    _A[8, 1] = 0
    _A[0, 2] = 0
    _A[1, 2] = 0
    _A[2, 2] = 1
    _A[3, 2] = 0
    _A[4, 2] = 0
    _A[5, 2] = 0
    _A[6, 2] = 0
    _A[7, 2] = 0
    _A[8, 2] = 0
    _A[0, 3] = 0
    _A[1, 3] = 0
    _A[2, 3] = 0
    _A[3, 3] = dt * (-_tmp0 * _tmp2 + _tmp0 * _tmp4) + 1
    _A[4, 3] = dt * (-_tmp5 - _tmp6)
    _A[5, 3] = dt * (-_tmp2 * _tmp8 + _tmp4 * _tmp8)
    _A[6, 3] = _tmp14 * (-_tmp1 * _tmp13 + _tmp10 * _tmp3)
    _A[7, 3] = _tmp14 * (-_tmp1 * _tmp16 - _tmp15 * _tmp3)
    _A[8, 3] = -1.0 * _tmp1 * _tmp17
    _A[0, 4] = 0
    _A[1, 4] = 0
    _A[2, 4] = 0
    _A[3, 4] = dt * (_tmp18 * _tmp5 + _tmp18 * _tmp6)
    _A[4, 4] = 1
    _A[5, 4] = dt * (_tmp20 * _tmp5 + _tmp20 * _tmp6)
    _A[6, 4] = _tmp12 * _tmp21
    _A[7, 4] = _tmp21 * _tmp9
    _A[8, 4] = -_tmp14 * _tmp19 * _tmp3
    _A[0, 5] = 0
    _A[1, 5] = 0
    _A[2, 5] = 0
    _A[3, 5] = 0
    _A[4, 5] = 0
    _A[5, 5] = 1
    _A[6, 5] = _tmp14 * (_tmp1 * _tmp15 - _tmp16 * _tmp3)
    _A[7, 5] = _tmp14 * (_tmp1 * _tmp10 + _tmp13 * _tmp3)
    _A[8, 5] = 0
    _A[0, 6] = dt
    _A[1, 6] = 0
    _A[2, 6] = 0
    _A[3, 6] = 0
    _A[4, 6] = 0
    _A[5, 6] = 0
    _A[6, 6] = 1
    _A[7, 6] = 0
    _A[8, 6] = 0
    _A[0, 7] = 0
    _A[1, 7] = dt
    _A[2, 7] = 0
    _A[3, 7] = 0
    _A[4, 7] = 0
    _A[5, 7] = 0
    _A[6, 7] = 0
    _A[7, 7] = 1
    _A[8, 7] = 0
    _A[0, 8] = 0
    _A[1, 8] = 0
    _A[2, 8] = dt
    _A[3, 8] = 0
    _A[4, 8] = 0
    _A[5, 8] = 0
    _A[6, 8] = 0
    _A[7, 8] = 0
    _A[8, 8] = 1
    return _A
