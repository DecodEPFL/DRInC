"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Simulation handler for different experiments. The main function is simulate.
Additional ultility functions are also provided.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from utils.data_structures import LinearSystem


def split_clm(phi, n_states, t_fir):
    """
    splits the closed loop map phi into its components at each time step.
    The initial format is [phi_w(0), ..., phi_w(t_fir), phi_v(0), ...,
    phi_v(t_fir)], where phi_w = [phi_xw, phi_uw] and phi_v = [phi_xv, phi_uv].

    :param phi: SLS closed loop map phi
    :param n_states: number of states of the system
    :param t_fir: length of the FIR SLS closed loop map filter
    :return: list of matrices, one for each delay component of phi
    """

    phi_xw = np.split(phi[:n_states, :n_states*t_fir], t_fir, axis=1)
    phi_xv = np.split(phi[:n_states, n_states*t_fir:], t_fir, axis=1)
    phi_uw = np.split(phi[n_states:, :n_states*t_fir], t_fir, axis=1)
    phi_uv = np.split(phi[n_states:, n_states*t_fir:], t_fir, axis=1)

    return [np.block([[phi_xw[i], phi_xv[i]], [phi_uw[i], phi_uv[i]]])
            for i in range(t_fir)]


def clm_to_dyn_ctrl(phi: np.ndarray, sys: LinearSystem):
    """
    Transform the closed loop map phi into a dynamical controller. This
    provides a practical implementation of the controller. The controller's
    parameters are in Canonical form using inputs 'y' from t-T to t and
    internal state 'd' from t-T to t-1:
    A = [0 I 0 0 0                              B = [0
         0 0 I 0 0                                   0
         0 0 0 I 0                                   0
         0 0 0 0 I                                   0
         0 -Phi[:n, :n*(T-1)]]                       -Phi[:n, -p*(T+1):]]
    C = [Phi[n:, :n*(T-1)] Phi[n:, :-p]C]      D = Phi[n:, -p*(T+1):]

    The inputs y from t-T to t are then merged in the states. This yields
    A = [0 I 0 0 0              0 0
         0 0 I 0 0              0 0
         0 0 0 I 0              0 0
         0 0 0 0 I              0 0
         0 -Phi[:n, :n*(T-1)]   0 0 -Phi[:n, -p*(T+1):-p]
         0 0 0 0 0              0 I
         0 0 0 0 0              0 0]
    B = [0 0 0 0 -Phi[:n, -p:].T 0 I].T
    C = [Phi[n:, :n*(T-1)] Phi[n:, :-p]C Phi[n:, -p*(T+1):]]

    :param phi: SLS closed loop map phi.
    :param sys: LinearSystem, system to control
    """
    _ctrl = LinearSystem()

    # Check that we don't already have a dynamical system
    if isinstance(phi, LinearSystem):
        # Avoid nonsense with imaginary d member
        _ctrl.a, _ctrl.b, _ctrl.c = phi.a, phi.b, phi.c
        return _ctrl

    # Get dimensions
    _n = sys.a.shape[0]
    _m = sys.b.shape[1] if sys.b is not None else 0
    _p = sys.c.shape[0] if sys.c is not None else 0
    sys.c = None if _p == 0 else sys.c
    sys.b = None if _m == 0 else sys.b
    _np = _n + _p
    _fir = int(phi.shape[1] / _np)

    # Check that the closed loop map and samples are compatible
    if _fir != phi.shape[1] / _np:
        raise ValueError(f"The closed loop map is not compatible"
                         f"with the system's dimensions")

    # Make dynamical system
    if _p == 0:
        _ctrl.a = np.block(
            [[np.zeros((_n * (_fir-2), _n)), np.eye(_n * (_fir-2))],
             [np.zeros((_n, _n)), -phi[:_n, _n:_n * (_fir-1)]]])
        _ctrl.b = np.block([[np.zeros((_n * (_fir - 2), _n))], [np.eye(_n)]])
        _ctrl.c = np.block([[phi[_n:, _n:_n * _fir]]])
    elif _m == 0:
        raise NotImplementedError("Observers are not implemented yet.")
    else:
        _ctrl.a = np.block(
            [[np.zeros((_n * (_fir-2), _n)), np.eye(_n * (_fir-2)),
              np.zeros((_n * (_fir-2), _p * _fir))],
             [np.zeros((_n, _n)), -phi[:_n, :_n * (_fir-2)],
              np.zeros((_n, _p)), -phi[:_n, -_p*_fir:-_p]],
             [np.zeros((_p * _fir, _n * (_fir-1))), np.eye(_p * _fir, k=_p)]])
        _ctrl.b = np.block([[np.zeros((_n * (_fir - 2), _p))], [phi[:_n, -_p:]],
                            [np.zeros((_p * (_fir - 1), _p))], [np.eye(_p)]])
        _ctrl.c = np.block([[phi[_n:, :_n * (_fir - 2)], phi[_n:, -_p:] @ sys.c,
                             phi[_n:, phi.shape[1]-_p*_fir:]]])

    return _ctrl


def simulate(phi, sys: LinearSystem,
             xis_profile: np.ndarray, x0: np.ndarray = None):
    """
    Simulates the closed loop system defined by the SLS closed loop map phi
    and the system sys, with the noise distribution given by the empirical
    distribution xis_profile.

    :param phi: SLS closed loop map phi if phi is an np.ndarray, or dynamical
        controller if phi is a LinearSystem.
    :param sys: LinearSystem, system to simulate
    :param xis_profile: empirical distribution over a finite horizon, each
        column is a sample.
    :param x0: initial states for t_fir time steps. If None, the origin is used.
    :return: x, u, y, the state, input, and output trajectories of
        the closed loop system. Each column is a trajectory corresponding to
        one sample.
    """
    # Short notations
    _ns = xis_profile.shape[1]
    _n = sys.a.shape[0]
    _m = sys.b.shape[1] if sys.b is not None else 0
    _p = sys.c.shape[0] if sys.c is not None else 0
    _np = _n + _p
    _t = int(xis_profile.shape[0] / _np)

    # Check that the system is square
    if sys.a.shape[0] != sys.a.shape[1]:
        raise ValueError("The system must be square.")

    # Check that the system is compatible
    if sys.b is not None and sys.b.shape[0] != sys.a.shape[0]:
        raise ValueError(f"Different number of rows in A and B. "
                         f"{sys.a.shape[0]} != {sys.b.shape[0]}")
    if sys.c is not None and sys.c.shape[1] != sys.a.shape[0]:
        raise ValueError(f"Different number of columns in A and C. "
                         f"{sys.a.shape[0]} != {sys.c.shape[1]}")

    # Transform closed loop map into dynamical controller if needed
    if not isinstance(phi, LinearSystem):
        _fir = int(phi.shape[1] / _np)
        # Implement CLM as dynamical system
        phi = clm_to_dyn_ctrl(phi, sys)
    else:
        _fir = 1

    if not hasattr(phi, 'd'):
        phi.d = None

    # Number of states in controller
    _nc = phi.a.shape[0]

    # Check that the controller is square
    if phi.a.shape[0] != phi.a.shape[1]:
        raise ValueError("The system must be square.")

    # Check that the controller is compatible
    if phi.b is not None:
        if phi.b.shape[0] != phi.a.shape[0]:
            raise ValueError(f"Different number of rows in A and B. "
                             f"{phi.a.shape[0]} != {phi.b.shape[0]}")
        if phi.d is not None:
            if phi.d.shape[1] != phi.b.shape[1]:
                raise ValueError(f"Different number of columns in B and D. "
                                 f"{phi.b.shape[1]} != {phi.d.shape[1]}")
    elif sys.c is not None:
        raise ValueError("The controller must have an input if the system has"
                         "an output.")
    if phi.c is not None:
        if phi.c.shape[1] != phi.a.shape[0]:
            raise ValueError(f"Different number of columns in A and C. "
                             f"{phi.a.shape[0]} != {phi.c.shape[1]}")
        if hasattr(phi, 'd') and phi.d is not None:
            if phi.d.shape[0] != phi.c.shape[0]:
                raise ValueError(f"Different number of rows in C and D. "
                                 f"{sys.c.shape[0]} != {sys.d.shape[0]}")
    elif sys.b is not None:
        raise ValueError("The controller must have an output if the system has"
                         "an input.")

    # Handle initial state
    if x0 is not None:
        if x0.shape[0] <= _n*_fir:
            raise ValueError(f"The initial state is not compatible"
                             f"with the system. Its dimension must be at least"
                             f"equal to the product of the number of delay"
                             f"terms in the closed loop map and the number of"
                             f"states. Got {x0.shape[0]} instead of {_n*_fir}.")
        else:
            x0 = x0[-_n*_fir:, :]
    else:
        x0 = np.zeros((_n * _fir, _ns))

    # Initial state of the controller, input and output
    # d = estimate disturbance
    d0 = np.vstack([np.vstack([
        (np.eye(_nc - _p * _fir, k=_n*(_fir-i-1))
         @ (x0[:_n * (_fir - 1), :] - x0[_n:, :]) if _fir > 1 else 0*x0),
        np.kron(np.eye(_fir, k=_fir-i-1), sys.c) @ x0 if sys.c is not None
        else np.zeros((0, _ns))]) for i in range(0, _fir)])
    y0 = np.kron(np.eye(_fir), sys.c) @ x0[-_n * _fir:, :] \
        if sys.c is not None else np.zeros((0, _ns))
    u0 = np.vstack((np.zeros(((_fir-1)*_m, _ns)), phi.c @ d0[-_nc:, :]
                    + (phi.d @ y0[-_p, :] if phi.d is not None else 0))) \
        if sys.b is not None else np.zeros((0, _ns))

    # Declare simulation variables
    [x, d, u, y] = [
        np.vstack((v0, np.zeros((_l * _t, _ns))))
        for _l, v0 in zip([_n, _nc, _m, _p], [x0, d0, u0, y0])
    ]

    # Simulate the closed loop system
    # xt+1 = A xt + B ut + wt
    # yt+1 = C xt+1 + vt+1
    # dt+1 = Ac dt + Bc [C xt+1, ..., C xt-T+1]
    # ut+1 = Cc dt + Dc [C xt+1, ..., C xt-T+1]
    for t in range(_t):
        tf = t + _fir
        x[_n*tf:_n*(tf+1), :] = sys.a @ x[_n*(tf-1):_n*tf, :] \
            + xis_profile[_n*t:_n*(t+1), :] \
            + (sys.b @ u[_m*(tf-1):_m*tf, :] if _m > 0 else 0)
        if sys.c is not None:
            y[_p*tf:_p*(tf+1), :] = sys.c @ x[_n*tf:_n*(tf+1), :] \
                + xis_profile[-_p*(_t-t):-_p*(_t-t-1) if t != _t-1 else None, :]

        y_c = y[_p*tf:_p*(tf+1), :] if _p > 0 else x[_n*tf:_n*(tf+1), :]
        d[_nc*tf:_nc*(tf+1), :] = phi.a @ d[_nc*(tf-1):_nc*tf, :] \
            + (phi.b @ y_c if phi.b is not None else 0)
        if sys.b is not None:
            u[_m*tf:_m*(tf+1), :] = phi.c @ d[_nc*tf:_nc*(tf+1), :]

    return x, u, y, d
