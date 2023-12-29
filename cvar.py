"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Closure generation for distributionally robust cvar constraints as defined in
"Distributionally Robust Infinite-horizon Control" (DRInC) by 
JS Brouillon et. al., 2023.

Copyright Jean-Sébastien Brouillon (2024)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def cvar_constraints(feasible_set, support):
    """
    This function generates the closure that defines the distributionally robust
    cvar constraints for a given feasible set and noise distribution support.
    The closure can be used afterward for various different optimization
    problems over the SLS closed loop map phi.

    :param feasible_set: Polytope, the feasible set of the optimization problem.
    :param support: Polytope, the support of the noise distribution.
    :return: closure with signature (phi) -> cons, where phi is the SLS closed
        loop map and cons is a list of linear matrix inequality constraints.
    """

    def mkcons(phi):
        cons = []
        return cons

    return mkcons
