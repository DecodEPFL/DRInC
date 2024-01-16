"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Dynamic controler synthesis for stationary LQG policy. This uses an optimal
Kalman filter and LQR controller, obtained by solving Riccatti equations.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from scipy.linalg import solve_discrete_are
from utils.data_structures import LinearSystem


def synthesize_lqg(sys: LinearSystem, cov_process: np.ndarray,
                   cov_measure: np.ndarray, weights: np.ndarray = None):
    """
    This function generates a dynamical system implementing the LQG controller.
    It uses the infinite-horizon LQR and LQE solutions to the Riccati equation.

    :param sys: LinearSystem to be controlled
    :param cov_process: np.ndarray, the covariance matrix of the process noise.
    :param cov_measure: np.ndarray, the covariance matrix of
        the measurement noise.
    :param weights: np.ndarray, the weight matrix of the LQR problem.
        Default is identity.
    :return: LinearSystem, the dynamical controller with internal states being
        the estimates of the system's state.
    """
    # Check what kind of system is at hand
    if sys.a is None:
        raise AttributeError("The system must have a state space matrix.")
    if sys.b is None or sys.c is None:
        raise AttributeError("The system must have an input and output matrix."
                             " For state-feedback or observer, use LQR/LQE.")

    # Check that the system is square
    if sys.a.shape[0] != sys.a.shape[1]:
        raise ValueError("The system must be square.")

    # Check that the system is compatible
    if sys.b.shape[0] != sys.a.shape[0] or sys.c.shape[1] != sys.a.shape[0]:
        raise ValueError("System dimensions are not compatible.")

    # Handle optional parameter
    weights = np.eye(sys.a.shape[0] + sys.b.shape[1]) if weights is not None \
        else weights @ weights.T  # original def is square root

    # Variables
    _n = sys.a.shape[0]
    _s = weights[:_n, _n:]
    _q = weights[:_n, :_n] + _s @ np.linalg.inv(weights[_n:, _n:]) @ _s.T
    _r = weights[_n:, _n:]
    _a = sys.a - sys.b @ np.linalg.inv(_r) @ _s.T

    # Controller design
    lqg = LinearSystem()

    # LQE / Kalman filter
    _pk = solve_discrete_are(sys.a.T, sys.c.T, cov_process, cov_measure)
    lqg.b = _pk @ sys.c.T @ np.linalg.inv(cov_measure + sys.c @ _pk @ sys.c.T)

    # LQR controller
    _pl = solve_discrete_are(_a, sys.b, _q, _r)
    lqg.c = -np.linalg.inv(_r + sys.b.T @ _pl @ sys.b) \
        @ (sys.b.T @ _pl @ sys.a + _s.T)

    # Closed loop system
    lqg.a = sys.a - lqg.b @ sys.c @ sys.a + sys.b @ lqg.c

    return lqg
