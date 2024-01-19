# DRInC
Distributionally Robust Infinite-horizon Controller synthesis

## Installation
Install numpy, scipy, cvxpy with mosek and pythorch. Then run main.py.

## Change parameters
The experiments/file double_integrator.py contains all the experiment-specific
parameters:
System, feasible set ([I, -I][x,u] ≤ [1,1] * feas_r), support (unit box),
FIR length, number of samples, and testing/training parameters.

The random distributions are explained in utils/distributions.py.

The cost function weights are defined at the beginning of main.py.

