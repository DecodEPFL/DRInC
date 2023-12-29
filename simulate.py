"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Simulation handler for different experiments. The main function is simulate.
Additional ultility functions are also provided.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np


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
