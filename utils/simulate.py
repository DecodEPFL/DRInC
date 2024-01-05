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


def simulate(phi: np.ndarray, sys: LinearSystem,
             xis_profile: np.ndarray, x0: np.ndarray = None):
    """
    Simulates the closed loop system defined by the SLS closed loop map phi
    and the system sys, with the noise distribution given by the empirical
    distribution xis_profile.
    :param phi: SLS closed loop map phi
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
    _fir = int(phi.shape[1] / (_n + _p))
    _t = int(xis_profile.shape[0] / (_n + _p))

    # Check that the system is square
    if sys.a.shape[0] != sys.a.shape[1]:
        raise ValueError("The system must be square.")

    # Check that the system is compatible
    if sys.b is not None and sys.b.shape[0] != sys.a.shape[0]:
        raise ValueError("Different number of rows in A and B.")
    if sys.c is not None and sys.c.shape[1] != sys.a.shape[0]:
        raise ValueError("Different number of columns in A and C.")

    # Check that the closed loop map and samples are compatible
    if _fir != phi.shape[1] / (_n + _p):
        raise ValueError(f"The closed loop map is not compatible"
                         f"with the system's dimensions")
    if _t != xis_profile.shape[0] / (_n + _p):
        raise ValueError(f"The samples are not compatible"
                         f"with the system's dimensions")

    # Initial state
    if x0 is None:
        x0 = np.zeros((phi.shape[1]-_n, 1))
    else:
        if x0.shape[0] != int(phi.shape[1]*_n / (_n+_p)):
            raise ValueError(f"The initial state is not compatible"
                             f"with the system. Its dimension must be equal to"
                             f"the product of the number of delay terms in the"
                             f"closed loop map and the number of states."
                             f"Got {x0.shape[0]} instead of "
                             f"{phi.shape[1]*_n / (_n+_p)}.")
        if len(x0.shape) == 1:
            xu0 = x0[:, None]
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
