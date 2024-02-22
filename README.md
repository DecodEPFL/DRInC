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


## Run
There are two different experiments you can run:
1. Keeping the same distribution class for the training and testing sets.
This is the default setting. To do this, call the program as 
`python main.py default [param]`, where param is the value of a varying
parameter of the distribution classes in chosen in the experiment module.
2. Using random distributions in the wasserstein ball for the testing set. To
do this, call the program as `python main.py random [radius]`, where radius is
the radius of the wasserstein ball.

You can run multiple tests at once by typing multiple values in the call, e.g.
`python main.py random "[0.05 0.05 0.02 0.02 0.01 0.01]"`
or `python main.py default "[0.5 1.0 1.5]"`.

## Examples from the paper
To generate the results from the paper, run the following commands. The first 
set of commands generates the results for the random distributions.

`python3 main.py random "sum([[i/50]*1000 for i in range(1,7)], [])"` <br />
`python3 plot_results.py bars`

The second set of commands for the run the experiment with parametric 
distributions.

`python3 main.py default "sum([[1.5 - i/10]*10 for i in range(11)], [])"` <br />
`python3 plot_results.py scatter`

The results are saved in the folder `results/` and are overwritten if the
experiment is run again. Note that the "Emp" controller is not implemented and
will return only zeros. The "Emp" controller presented in the paper is obtained
by running DRInC with a radius of 0.0001, and adding the results manually.
