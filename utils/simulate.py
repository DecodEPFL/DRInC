"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Simulation handler for different experiments. The main function is simulate.
Additional ultility functions are also provided.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np


def simulate(phi, sys, xis_profile, x0=None):
    """
    Simulates the closed loop system defined by the SLS closed loop map phi
    and the system sys, with the noise distribution given by the empirical
    distribution xis_profile.
    :param phi: SLS closed loop map phi
    :param sys: LinearSystem, system to simulate
    :param xis_profile: empirical distribution over a finite horizon, each
        column is a sample.
    :param x0: initial state for t_fir time steps, if None, the origin is used.
    :return: x, u, y, e, the state, input, output, and error trajectories of
        the closed loop system.
    """


def split_clm(phi, n_states, t_fir):
    """
    splits the closed loop map phi into its components.

    :param phi: SLS closed loop map phi
    :param n_states: number of states of the system
    :param t_fir: length of the FIR SLS closed loop map filter
    :return: list of matrices, one for each delay component of phi
    """

    phi_xw = np.split(phi[:n_states, :n_states*t_fir], t_fir, axis=1)
    phi_uw = np.split(phi[:n_states, n_states*t_fir:], t_fir, axis=1)
    phi_xv = np.split(phi[n_states:, :n_states*t_fir], t_fir, axis=1)
    phi_uv = np.split(phi[n_states:, n_states*t_fir:], t_fir, axis=1)

    return reversed([np.block([[phi_xw[i], phi_uw[i]], [phi_xv[i], phi_uv[i]]])
                     for i in range(t_fir)])
