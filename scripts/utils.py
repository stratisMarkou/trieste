from typing import *
import os
import shutil

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

gp_objectives = list(objective_kernels.keys())


def arg_summary(args: dict):
    return "\n".join([f"{k:<32} | {v:<64}" for k, v in args.__dict__.items()])


def make_gp_objective(
        kernel: str,
        num_fourier_components: int,
        dim: int,
        linear_stddev: float = 1e-6,
        num_initial_points: int = 10000,
        num_optimization_runs: int = 1000,
        dtype: tf.DType = tf.float64,
    ):

    assert kernel in objective_kernels

    search_space = trieste.space.Box(lower=dim*[-1.], upper=dim*[1.])

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
    reshaped_objective = lambda x: -objective(x[:, 0, :])

    continuous_optimizer = generate_continuous_optimizer(
        num_initial_samples=num_initial_points,
        num_optimization_runs=num_optimization_runs,
    )

    minimizer = continuous_optimizer(
        space=search_space,
        target_func=reshaped_objective,
    )
    minimum = objective(minimizer)

    return objective, minimum, search_space


def set_up_logging(args, summary):

    # Make directory to save results
    params = f"seed-{args.search_seed}"

    if not (args.acquisition == "ei"):
        params = params + f"_batch_size-{args.batch_size}"

    if "mcei" in args.acquisition:
        params = params + f"_num_mcei_samples-{args.num_mcei_samples}"

    if args.objective in gp_objectives:
        objective_name = f"{args.objective}_" + \
                         f"{args.objective_dimension}_" + \
                         f"{args.objective_seed}"

    else:
        objective_name = args.objective

    path = os.path.join(
        args.save_dir,
        args.acquisition,
        objective_name,
        params,
    )

    os.makedirs(path, exist_ok=True)
    with open(f"{path}/arguments.txt", "w") as file:
        file.write(summary)
        file.close()

    if os.path.exists(f"{path}/log-regret.txt"):
        os.remove(f"{path}/log-regret.txt")

    if os.path.exists(f"{path}/log"):
        shutil.rmtree(f"{path}/log")

    summary_writer = tf.summary.create_file_writer(f"{path}/log")
    trieste.logging.set_tensorboard_writer(summary_writer)

    return path
