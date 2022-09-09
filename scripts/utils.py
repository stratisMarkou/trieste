from typing import *

import gpflow
import trieste
import tensorflow as tf

from trieste.acquisition.optimizer import generate_continuous_optimizer
from gpflux.layers.basis_functions.fourier_features import RandomFourierFeaturesCosine


objective_kernels = {
    "eq": gpflow.kernels.RBF,
    "matern12": gpflow.kernels.Matern12,
    "matern32": gpflow.kernels.Matern32,
    "matern52": gpflow.kernels.Matern52,
}

def arg_summary(args: dict):
    return "\n".join([f"{k:<32} | {v:<64}" for k, v in args.__dict__.items()])

def make_gp_objective(
        kernel: str,
        num_fourier_components: int,
        search_space: trieste.space.SearchSpace,
        linear_stddev: float = 1e-6,
        num_initial_points: int = 10000,
        num_optimization_runs: int = 1000,
        dtype: tf.DType = tf.float64,
    ):

    assert kernel in objective_kernels

    D = search_space.dimension

    kernel = objective_kernels[kernel]()

    rff = RandomFourierFeaturesCosine(
        kernel=kernel,
        n_components=num_fourier_components,
        dtype=dtype,
    )

    weights = tf.random.normal(shape=(num_fourier_components,), dtype=dtype)
    linear_weights = linear_stddev * tf.random.normal(shape=(D,), dtype=dtype)

    objective = lambda x: tf.linalg.matvec(a=rff(x), b=weights)[:, None] + \
                          tf.reduce_sum(linear_weights[None, :] * x, axis=1)[:, None]
    reshaped_objective = lambda x: - objective(x[:, 0, :])

    continuous_optimizer = generate_continuous_optimizer(
        num_initial_samples=num_initial_points,
        num_optimization_runs=num_optimization_runs,
    )

    minimizer = continuous_optimizer(
        space=search_space,
        target_func=reshaped_objective,
    )
    minimum = objective(minimizer)

    return objective, minimum
