import numpy as np
import cvxpy as cp
from cvar import *
from data_structures import LinearSystem, Polytope
from cvar import cvar_constraints


def test_cvar_constraints(verbose=False):
    """
    tests the cvar constraints generation for a random closed loop map.
    :param verbose: bool, if True, prints the optimization verbose.
    """
    np.random.seed(123)

    # Parameters
    _m, _n, _p = 2, 3, 2
    t_fir = 4
    radius = 0.1
    p_level = 5e-2
    _ntrain, _ntest = 10, 100

    # Generate a random box feasible set
    fset = Polytope()
    fset.h = np.vstack((np.eye(_n + _m),
                        -np.eye(_n + _m)))
    fset.g = np.random.uniform(0, 10,
                               (2 * (_n + _m), 1))
    # Uniform distribution with box support from -1 to 1
    support = Polytope()
    support.h = np.vstack((np.eye((_n + _p) * t_fir),
                           -np.eye((_n + _p) * t_fir)))
    support.g = np.ones((2 * (_n + _p) * t_fir, 1))

    # Generate the constraints
    mkcons = cvar_constraints(fset, support, radius, p_level)

    # Generate a random closed loop map
    phi = cp.Variable(((_n + _m), (_n + _p) * t_fir))
    phi_tar = np.random.randn((_n + _m), (_n + _p) * t_fir)

    # Generate random noise realizations
    train_xis = np.random.uniform(-1, 1,
                                  ((_n + _p) * t_fir, _ntrain))
    test_xis = np.random.uniform(-1, 1,
                                 ((_n + _p) * t_fir, _ntest))

    # Try to be random but still satisfy constraints
    cp.Problem(cp.Minimize(cp.norm(phi - phi_tar, 'fro')),
               mkcons(phi, train_xis)).solve(verbose=verbose)

    # Compute expected amount of constraint violations
    violations_tar = np.max(fset.h @ phi_tar @ test_xis - fset.g, axis=0)
    exp_viol_tar = np.mean(np.sort(violations_tar)[-int(p_level * _ntest):])
    violations = np.max(fset.h @ phi @ test_xis - fset.g, axis=0)
    exp_viol = np.mean(np.sort(violations)[-int(p_level * _ntest):])

    # Check constraints
    assert(exp_viol <= 1e-8)
    
    if verbose:
        print(f"Expected constraint violation of target: {exp_viol_tar}")
        print(f"Constraint violation of result: {exp_viol}")
        print("All tests passed")


# Press the green button in the gutter to run the test.
if __name__ == '__main__':
    np.set_printoptions(linewidth=120, precision=2)
    # for k in range(10):
    test_cvar_constraints(True)
