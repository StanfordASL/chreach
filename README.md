# Convex Hulls of Reachable Sets

## About

Code for our work on convex hull reachability analysis (T. Lew, R. Bonalli, M. Pavone, "Convex Hulls of Reachable Sets", available at [https://arxiv.org/abs/2303.17674](https://arxiv.org/abs/2303.17674), 2024).

![continuous_time_reachability](/results/continuous_time_reachability.jpg)

## Using this code
To reproduce
* the spacecraft control results, run
``python scripts/spacecraft.py``
* the Dubins car result with rectangular disturbance sets, run
``python dubins_rectangle.py``
* the Dubins car result with non-invertible disturbance dynamics, run
``python dubins_non_invertible.py``
* the neural feedback loop analysis results, see https://github.com/StanfordASL/nn_robustness_analysis/tree/disturbances_and_initial_states

Other examples can be found in the ``scripts/`` and ``tests/`` folder.

## Setup
This code was tested with Python 3.8.0 on Ubuntu 18.04.6.

We recommend installing the package in a virtual environment. First, run 
```bash
  python -m venv ./venv
  source venv/bin/activate
``` 
Then, all dependencies (numpy, scipy, jax, osqp, and matplotlib) can be installed by running 
```
  pip install -r requirements.txt
```
and the package can be installed by running
```
  pip install -e .
```

## Testing
The following command should run successfully:
``
  python -m pytest
``
