"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Closure generation for achievability constraints as defined in
"System Level Synthesis" (SLS) by J Anderson et. al., 2019.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from utils.data_structures import LinearSystem


def achievability_constraints(sys: LinearSystem, t_fir: int):
    """
    This function generates the closure that defines the achievability
    constraints for the system sys. The closure can be used afterward for
    various different optimization problems over the SLS closed loop map phi.

    :param sys: LinearSystem for which the constraints apply.
    :param t_fir: int > 0, length of the FIR SLS closed loop map filter.
    :return: closure with signature (phi) -> cons, where phi is the SLS closed
        loop map and cons is a list of linear matrix equality constraints.
    """

    # Check that the horizon is positive
    if t_fir <= 0:
        raise ValueError("The horizon must be positive.")

    # Check what kind of system is at hand
    if sys.a is None:
        raise AttributeError("The system must have a state space matrix.")
    if sys.b is None or sys.c is None:
        raise AttributeError("The system must have an input or output matrix.")

    # Check that the system is square
    if sys.a.shape[0] != sys.a.shape[1]:
        raise ValueError("The system must be square.")

    # Check that the system is compatible
    if sys.b is not None and sys.b.shape[0] != sys.a.shape[0]:
        raise ValueError("Different number of rows in A and B.")
    if sys.c is not None and sys.c.shape[1] != sys.a.shape[0]:
        raise ValueError("Different number of columns in A and C.")

    # Short notations
    _t = t_fir
    _n = sys.a.shape[0]
    _m = 0 if sys.b is None else sys.b.shape[1]
    _p = 0 if sys.c is None else sys.c.shape[0]
    _k = np.kron
    _i = np.eye

    def blkdiag(_x, _y):  # Numpy does not handle "None" in block matrices
        return np.block([[_x, np.zeros((_x.shape[0], _y.shape[1]))],
                         [np.zeros((_y.shape[0], _x.shape[1])), _y]])

    # Handy matrices
    if sys.b is not None:
        _io = np.hstack((_i(_n), 0 * sys.b))
    _zp = np.hstack((np.eye(_t), np.zeros((_t, 1))))
    _zm = np.hstack((np.zeros((_t, 1)), np.eye(_t)))

    # Define the closures for state feedback, observers, and output feedback
    if sys.c is None:  # TODO: Not tested !!!
        def mkcons(phi):
            cons = [_io @ phi @ _k(_zm, _i(_n))
                    == np.hstack((sys.a, sys.b)) @ phi @ _k(_zp, _i(_n))
                    + _k(_zp[-1, :], _i(_n))]
            return cons
    elif sys.b is None:  # TODO: Not tested !!!
        def mkcons(phi):
            cons = [phi @ np.vstack((_k(_zm, _i(_n)), _k(_zm, 0 * sys.c)))
                    == phi @ np.vstack((_k(_zp, sys.a), _k(_zp, sys.c)))
                    + _k(_zp[-1, :], _i(_n))]
            return cons
    else:
        def mkcons(phi):
            cons = [_io @ phi @
                    blkdiag(_k(_zm, _i(_n)), _k(_zm, _i(_p)))
                    == np.hstack((sys.a, sys.b)) @ phi @
                    blkdiag(_k(_zp, _i(_n)), _k(_zp, _i(_p)))
                    + np.hstack((_k(_zp[[-1], :], _i(_n)),
                                 _k(_zp[[-1], :], 0 * sys.c.T)))]

            cons += [phi @ np.vstack((_k(_zm, _i(_n)), _k(_zm, 0 * sys.c)))
                     == phi @ np.vstack((_k(_zp, sys.a), _k(_zp, sys.c)))
                     + np.vstack((_k(_zp[[-1], :], _i(_n)),
                                  _k(_zp[[-1], :], 0 * sys.c)))]
            return cons

    return mkcons
