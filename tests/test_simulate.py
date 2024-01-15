import numpy as np
import cvxpy as cp
from utils.simulate import split_clm, simulate
from utils.data_structures import LinearSystem
from achievability import achievability_constraints


def test_simulate(verbose=False):
    """
    tests the achievability constraints generation for a random closed loop map.
    :param verbose: bool, if True, prints the optimization verbose.
    """
    _tf, _t = 5, 10
    _m, _n, _p = 1, 2, 1
    sys = LinearSystem()
    sys.a, sys.b, sys.c = np.random.randn(_n, _n), \
        np.random.randn(_n, _m), np.random.randn(_p, _n)
    xi = np.vstack((np.random.randn(_t*_n, 1), np.random.randn(_t*_p, 1)))

    xi0_xi = np.vstack(
        (np.zeros((_n * (_tf-1), 1)), xi[:_n * _t, :],
         np.zeros((_m * (_tf-1), 1)), xi[-_p * _t:, :]))

    phi = cp.Variable((_n+_p, (_n+_p) * _tf))
    cp.Problem(cp.Minimize(cp.norm(phi, 'fro')),
               achievability_constraints(sys, _tf)(phi)).solve(verbose=verbose)
    x, u, y, _ = simulate(phi.value, sys, xi)
    xu = np.vstack((x[_n*_tf:, :], u[_tf:, :]))

    xu_gt = np.zeros_like(xu)
    for t in range(0, _t):
        xii = (xi0_xi[_n*(t+1):_n*(t+_tf+1), :],
               xi0_xi[-_p*(_t-t+_tf-1):-_p*(_t-t-1) if t != _t-1 else None, :])

        xu_gt[[_n*t, _n*t+1, _p*(t-_t)], :] = (phi.value @ np.vstack(xii))

    if verbose:
        np.set_printoptions(precision=2, suppress=True)
        print("states:")
        print(xu[:_n*_t, :])
        print("states ground truth:")
        print(xu_gt[:_n*_t, :])
        print("inputs:")
        print(xu[_n*_t:, :])
        print("inputs ground truth:")
        print(xu_gt[_n*_t:, :])

    np.testing.assert_allclose(xu, xu_gt, rtol=0, atol=1e-7)

    if verbose:
        print("All tests passed")


# Press the green button in the gutter to run the test.
if __name__ == '__main__':
    # Not using generators for unit tests
    np.random.seed(123)

    test_simulate(False)
