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
    phi_uw = np.split(phi[:n_states, n_states*t_fir:], t_fir, axis=1)
    phi_xv = np.split(phi[n_states:, :n_states*t_fir], t_fir, axis=1)
    phi_uv = np.split(phi[n_states:, n_states*t_fir:], t_fir, axis=1)

    return [np.block([[phi_xw[i], phi_uw[i]], [phi_xv[i], phi_uv[i]]])
            for i in range(t_fir)]


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
        raise ValueError("Different number of rows in A and B.")
    if sys.c is not None and sys.c.shape[1] != sys.a.shape[0]:
        raise ValueError("Different number of columns in A and C.")

    # Transform closed loop map into dynamical controller
    if not isinstance(phi, LinearSystem):
        _ctrl = LinearSystem()
        _fir = int(phi.shape[1] / _np)
        _phi_i = phi

        # Check that the closed loop map and samples are compatible
        if _fir != phi.shape[1] / _np:
            raise ValueError(f"The closed loop map is not compatible"
                             f"with the system's dimensions")
        if _t != xis_profile.shape[0] / _np:
            raise ValueError(f"The samples are not compatible"
                             f"with the system's dimensions")

        """
        Canonical form: inputs 'y' from t-T to t
        and internal state 'd' from t-T to t-1
        A = [0 I 0 0 0                              B = [0
             0 0 I 0 0                                   0
             0 0 0 I 0                                   0
             0 0 0 0 I                                   0
             0 -Phi[:n, :n*(T-1)]]                       -Phi[:n, -p*(T+1):]]
        C = [Phi[n:, :n*(T-1)] Phi[n:, :-p] C]      D = Phi[n:, -p*(T+1):]
        """
        _ctrl.a = np.block([[np.zeros((_n*(_fir-2), _n)), np.eye(_n*(_fir-2))],
                            [np.zeros((_n, _n)), -phi[:_n, :_n*(_fir-2)]]])
        _ctrl.b = np.block([[np.zeros((_n*(_fir-2), _p*_fir))],
                            [-phi[:_n, -_p*_fir:]]])
        _ctrl.c = np.block([[phi[_n:, :_n*(_fir-2)], phi[_n:, -_p:]
                             @ (sys.c if sys.c is not None else 1)]])
        _ctrl.d = phi[_n:, -_p*_fir:]
        phi = _ctrl

    else:
        _fir = 1
        if not hasattr(phi, 'd'):
            phi.d = None

    # States of controller
    _nc = phi.a.shape[0]

    # Check that the controller is square
    if phi.a.shape[0] != phi.a.shape[1]:
        raise ValueError("The system must be square.")

    # Check that the controller is compatible
    if phi.b is not None:
        if phi.b.shape[0] != phi.a.shape[0]:
            raise ValueError("Different number of rows in A and B.")
        if phi.d is not None:
            if phi.d.shape[1] != phi.b.shape[1]:
                raise ValueError("Different number of columns in B and D.")
    elif sys.c is not None:
        raise ValueError("The controller must have an input if the system has"
                         "an output.")
    if phi.c is not None:
        if phi.c.shape[1] != phi.a.shape[0]:
            raise ValueError("Different number of columns in A and C.")
        if hasattr(phi, 'd') and phi.d is not None:
            if phi.d.shape[0] != phi.c.shape[0]:
                raise ValueError("Different number of rows in C and D.")
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
    d0 = np.vstack([np.eye(_nc, k=_n*(_fir-i-1)) for i in range(0, _fir)]) \
        @ ((x0[:_n * (_fir - 1), :] - x0[_n:, :]) if _fir > 1 else 0*x0)
    y0 = np.kron(np.eye(_fir), sys.c) @ x0 \
        if sys.c is not None else np.zeros((0, _ns))
    u0 = np.vstack((np.zeros(((_fir-1)*_m, _ns)), phi.c @ d0[-_nc:, :]
                    + (phi.d @ y0 if phi.d is not None else 0))) \
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
            + (sys.b @ u[_m*(tf-1):_m*tf, :] if sys.b is not None else 0)
        if sys.c is not None:
            y[_p*tf:_p*(tf+1), :] = sys.c @ x[_n*tf:_n*(tf+1), :] \
                + xis_profile[-_p*(_t-t):-_p*(_t-t-1) if t != _t-1 else None, :]

        d[_nc*tf:_nc*(tf+1), :] = phi.a @ d[_nc*(tf-1):_nc*tf, :] \
            + (phi.b @ y[_p*t+_p:_p*(tf+1), :] if phi.b is not None else 0)
        if sys.b is not None:
            u[_m*tf:_m*(tf+1), :] = phi.c @ d[_nc*tf:_nc*(tf+1), :] \
                + (phi.d @ y[_p*t+_p:_p*(tf+1), :] if phi.d is not None else 0)

    return x, u, y, d


def _simulate_dyn(phi: np.ndarray, sys: LinearSystem,
                  xis_profile: np.ndarray, x0: np.ndarray = None):
    """
    Implementation of simulate for the closed loop map case.
    """
    # Short notations
    _ns = xis_profile.shape[1]
    _n = sys.a.shape[0]
    _m = sys.b.shape[1] if sys.b is not None else 0
    _p = sys.c.shape[0] if sys.c is not None else 0
    _np = _n + _p
    _fir = int(phi.shape[1] / (_n + _p))
    _t = int(xis_profile.shape[0] / (_n + _p))

    # Check that the closed loop map and samples are compatible
    if _fir != phi.shape[1] / (_n + _p):
        raise ValueError(f"The closed loop map is not compatible"
                         f"with the system's dimensions")
    if _t != xis_profile.shape[0] / (_n + _p):
        raise ValueError(f"The samples are not compatible"
                         f"with the system's dimensions")

    # Initial state
    if x0 is None:
        x0 = np.zeros((int(phi.shape[1]*_n / (_n+_p)), 1))
    else:
        if x0.shape[0] != int(phi.shape[1]*_n / (_n+_p)):
            raise ValueError(f"The initial state is not compatible"
                             f"with the system. Its dimension must be equal to"
                             f"the product of the number of delay terms in the"
                             f"closed loop map and the number of states."
                             f"Got {x0.shape[0]} instead of "
                             f"{phi.shape[1]*_n / (_n+_p)}.")
        if len(x0.shape) == 1:
            x0 = x0[:, None]
        if x0.shape[1] != 1:
            raise ValueError(f"Only one initial state and input trajectory"
                             f"can be provided, got {x0.shape[1]} instead.")
    # We do not check that xu0 is compatible with the system's dynamics

    # Change the closed loop map structure to slide better on longer horizons
    phi = np.hstack(split_clm(phi, _n, _fir))

    # Simulate the closed loop system
    xu = np.zeros(((_n+_p)*(_t-_fir), _ns))
    for i, xi in enumerate(xis_profile.T):
        for t in range(_t-_fir):
            # The time for xu and xi are shifted by _fir
            xu[t*_np:(t+1)*_np, i] = phi @ xi[t*_np:(t+_fir)*_np]

    # Extract the states and inputs
    x = np.vstack((x0 @ np.ones((1, _ns)), np.zeros((_n*(_t - _fir), _ns))))
    u = np.zeros((_m*_t, _ns)) if sys.b is not None else None
    for t in range(_fir, _t):
        tf = t - _fir
        x[t*_n:(t+1)*_n, :] = xu[tf*_np:tf*_np+_n, :]
        if sys.b is not None:
            u[t*_m:(t+1)*_m, :] = xu[tf*_np+_n:(tf+1)*_np, :]

    # Extract the outputs
    if sys.c is not None:
        y = np.zeros((_p*_t, _ns))
        for t in range(_t):
            y[t*_p:(t+1)*_p, :] = sys.c @ x[t*_n:(t+1)*_n, :]
    else:
        y = None

    return x, u, y
