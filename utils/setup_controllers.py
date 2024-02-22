"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Gather controller synthesis functions and return closures for each controller.
This allows to compare all benchamrks to DRInC in a unified way.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from drinc import synthesize_drinc
from utils.data_structures import LinearSystem, Polytope
from utils.distributions import get_distribution
from benchmarks.filtered_lqg import synthesize_auglqg
from benchmarks.lqg import synthesize_lqg
# Try to import robust and drlqg, if not available, skip
# (they require the pycddlib and pytorch packages, respectively)
from benchmarks.robust import synthesize_robust
try:
    from benchmarks.drlqg import drlqg_covariances
except ImportError:
    drlqg_covariances = None


def get_controllers(t_fir: int, radius: float, p_level: float,
                    sys: LinearSystem, fset: Polytope, support: Polytope,
                    verbose=False):
    """
    Sets the DRInC, robust, empirical, LQG, and DR-LQG controllers.
    returns closures with signature (xis, weights) -> phi, where phi is the SLS
    closed loop map, xis are the samples of the empirical distribution that
    the controllers are optimized for (one column per sample), and weights
    is the matrix square root of the weights W of the control cost
    [x_t, u_t]^T W [x_t, u_t].

    :param t_fir: int > 0, length of the FIR SLS closed loop map filter.
    :param radius: Radius of the Wasserstein ball
    :param p_level: Probability level of the cvar constraints
    :param sys: LinearSystem to control.
    :param fset: Polytope, the feasible set of the optimization problem.
    :param support: Polytope, the support of the noise distribution.
    :param verbose: bool, if True, prints the optimization verbose.
    :return: list of closures for each controller in controller_names.
    """
    _n, _p = sys.a.shape[0], sys.c.shape[0]

    # Obtain drinc closure
    drinc = synthesize_drinc(sys, t_fir, fset, support,
                             radius, p_level, radius, None, verbose)

    # Obtain H2 controller closure
    emp = synthesize_auglqg(sys, t_fir, fset, verbose)

    # Obtain robust closure, cut low probability part of support for feasibility
    rob_support = Polytope()
    rob_support.h, rob_support.g = support.h, support.g - 0.1*np.sign(support.g)
    rob = synthesize_robust(sys, t_fir, fset, rob_support, verbose)

    # Make lqg closure for compatibility. Use empirical covariances
    def lqg(xis, weights=None):
        p_cov = np.cov(np.hstack([xis[_n*i:_n*(i+1), :] for i in range(t_fir)]))
        m_cov = np.cov(np.hstack([xis[_n*t_fir + _p*i:_n*t_fir + _p*(i+1),
                                      :] for i in range(t_fir)]))
        return synthesize_lqg(sys, p_cov, m_cov, weights)

    # Skip if pythorch is not installed
    if drlqg_covariances is not None:
        # Make DR-LQG closure as LQG with worst-case variances
        def drlqg(xis, weights=None):
            # Center distribution variances
            _w = np.cov(np.hstack([xis[_n*i:_n*(i+1), :]
                                   for i in range(t_fir)]))
            _v = np.cov(np.hstack([xis[_n*t_fir + _p*i:_n*t_fir + _p*(i+1), :]
                                   for i in range(t_fir)]))

            # Worst case computation, t_fir = 1 as for LQG
            _, _, p_cov, m_cov = drlqg_covariances(sys, 1, _w, _v, radius)
            # Remove initial state variance (same as process noise)
            p_cov = p_cov[-_n:, -_n:]

            # LQG with worst case variances
            return synthesize_lqg(sys, p_cov, m_cov, weights)
    else:
        print("Warning: Install pytorch to enable DR-LQG. Skipping...")
        drlqg = None

    return {"DRInC": drinc, "Emp": None, "Robust": rob, "LQG": lqg,
            "DR-LQG": drlqg}


