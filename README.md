# Exact characterization of the convex hulls of reachable sets

## About

Code for our work on convex hull reachability analysis under disturbances (T. Lew, R. Bonalli, M. Pavone, ["Exact characterization of the convex hulls of reachable sets", available at [https://arxiv.org/abs/2302.13970](https://arxiv.org/abs/2302.13970), 2023).
* For a simple example, run
``python example.py``
* The spacecraft results can be reproduced by running
``python spacecraft.py``
* To reproduce the neural feedback loop analysis results, see https://github.com/StanfordASL/nn_robustness_analysis/tree/disturbances
* For further experiments using RandUP, which inspired the design of Algorithm 1 (which is basically RandUP + leveraging the geometric structure of the problem see Theorem 1 of the paper), please refer to the following repositories:
    * For results with latest 2nd order bounds: https://github.com/StanfordASL/convex-hull-estimation
    * For neural network verification: https://github.com/StanfordASL/nn_robustness_analysis. 
    * Code for L4DC 2022 publication: https://github.com/StanfordASL/RandUP.
    * Code for CoRL 2020 publication: https://github.com/StanfordASL/UP.
    * Some hardware results: https://youtu.be/sDkblTwPuEg.

## Setup
This code was tested with Python 3.7.16.

All dependencies (i.e., numpy, scipy, jax, and matplotlib) can be installed by running 
``
  pip install -r requirements.txt
``