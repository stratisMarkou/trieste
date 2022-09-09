import os
import sys
sys.path.append("./scripts")

import argparse

import gpflow
import numpy as np

import trieste
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.acquisition.optimizer import generate_continuous_optimizer

from trieste.acquisition.rule import (
    RandomSampling,
    EfficientGlobalOptimization,
)

from trieste.models.gpflow.builders import (
    build_gpr,
)

from trieste.acquisition.function import (
    ExpectedImprovement,
    BatchMonteCarloExpectedImprovement,
    AnnealedBatchMonteCarloExpectedImprovement,
    GreedyContinuousThompsonSampling,
)

from trieste.objectives.utils import mk_observer
from trieste.objectives.single_objectives import *

from utils import make_gp_objective, arg_summary


standard_objectives = {
    "scaled_branin": (
        scaled_branin,
        SCALED_BRANIN_MINIMUM,
        BRANIN_SEARCH_SPACE,
    ),
    "shekel_4": (
        shekel_4,
        SHEKEL_4_MINIMUM,
        SHEKEL_4_SEARCH_SPACE,
    ),
    "hartmann_3": (
        hartmann_3,
        HARTMANN_3_MINIMUM,
        HARTMANN_3_SEARCH_SPACE,
    ),
    "hartmann_6": (
        hartmann_6,
        HARTMANN_6_MINIMUM,
        HARTMANN_6_SEARCH_SPACE,
    ),
}

gp_objectives = [
    "eq",
    "matern32",
    "matern52",
]


def build_model(
        dataset: trieste.data.Dataset,
        trainable_noise: bool,
        kernel=None
    ) -> GaussianProcessRegression:
    """
    :param dataset:
    :param trainable_noise:
    :param kernel:
    :return:
    """

    variance = tf.math.reduce_variance(dataset.observations)

    if kernel is None:
        kernel = gpflow.kernels.Matern52(variance=variance)

    else:
        kernel = kernel(variance)

    gpr = gpflow.models.GPR(dataset.astuple(), kernel, noise_variance=1e-4)

    if not trainable_noise:
        gpflow.set_trainable(gpr.likelihood, False)

    return GaussianProcessRegression(gpr)


def log10_regret(queries: tf.Tensor, minimum: tf.Tensor) -> tf.Tensor:
    """
    :param queries:
    :param minimum:
    :return:
    """
    regret = tf.reshape(tf.reduce_min(queries) - minimum, shape=())
    return tf.math.log(regret) / tf.cast(tf.math.log(10.), dtype=regret.dtype)


def save_results(iteration, path, optimizer, minimum):

    observations = optimizer.to_result().try_get_final_dataset().observations
    log_regret = log10_regret(queries=observations, minimum=minimum)

    with open(f"{path}/log-regret.txt", "a") as file:
        file.write(f"{iteration}, {observations.shape[0]}, {log_regret}\n")
        file.close()

    return log_regret


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "acquisition",
        type=str,
        choices=[
            "random",
            "thompson",
            "ei",
            "mcei",
            "amcei",
        ],
        help="Which acquisition strategy to use",
    )

    parser.add_argument(
        "objective",
        type=str,
        choices=list(standard_objectives.keys()) + gp_objectives,
    )

    parser.add_argument(
        "-linear_stddev",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "-search_seed",
        type=int,
    )

    parser.add_argument(
        "-batch_size",
        type=int,
    )

    parser.add_argument(
        "-num_batches",
        type=int,
    )

    parser.add_argument(
        "--objective_seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num_initial_designs",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--num_ei_acquisitions",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--num_optimization_runs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--objective_dimension",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=5.,
    )

    parser.add_argument(
        "--objective_fourier_components",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--trainable_noise",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--num_mcei_samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="experiments",
    )

    args = parser.parse_args()

    summary = arg_summary(args)
    print(summary)

    assert not (args.objective in gp_objectives and args.objective_dimension == -1)

    # Make directory to save results
    params = f"seed-{args.search_seed}"

    if not (args.acquisition == "ei"):
        params = params + f"_batch_size-{args.batch_size}"

    if "mcei" in args.acquisition:
        params = params + f"_num_mcei_samples-{args.num_mcei_samples}"

    if args.objective in gp_objectives:
        objective_name = f"{args.objective}_{args.objective_dimension}_{args.objective_seed}"

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

    # Set objective
    if args.objective in standard_objectives:
        objective, minimum, search_space = standard_objectives[args.objective]

    else:

        search_space = trieste.space.Box(
            lower=args.objective_dimension*[-1.],
            upper=args.objective_dimension*[1.],
        )

        tf.random.set_seed(args.objective_seed)
        np.random.seed(args.objective_seed)

        objective, minimum = make_gp_objective(
            kernel=args.objective,
            num_fourier_components=args.objective_fourier_components,
            search_space=search_space,
            linear_stddev=args.linear_stddev,
        )

    # Set search seed
    tf.random.set_seed(args.search_seed)
    np.random.seed(args.search_seed)

    # Observe initial points
    D = int(search_space.dimension)
    num_initial_points = 2 * D + 2
    initial_query_points = search_space.sample(num_initial_points)
    observer = mk_observer(objective)
    initial_dataset = observer(initial_query_points)

    model = build_gpr(
        data=initial_dataset,
        search_space=search_space,
        likelihood_variance=1e-5,
    )

    model = GaussianProcessRegression(model)

    # Create acquisition rule
    if args.acquisition == "random":
        rule = RandomSampling(num_query_points=args.batch_size)
        num_bo_steps = args.num_batches

    else:

        if args.acquisition == "thompson":
            acquisition_function = GreedyContinuousThompsonSampling()
            num_bo_steps = args.num_batches
            batch_size = args.batch_size

        elif args.acquisition == "ei":
            acquisition_function = ExpectedImprovement()
            num_bo_steps = args.batch_size * args.num_batches
            batch_size = 1

        elif args.acquisition == "mcei":
            acquisition_function = BatchMonteCarloExpectedImprovement(
                sample_size=args.num_mcei_samples,
            )
            num_bo_steps = args.num_batches
            batch_size = args.batch_size

        elif args.acquisition == "amcei":
            acquisition_function = AnnealedBatchMonteCarloExpectedImprovement(
                beta=args.beta,
                sample_size=args.num_mcei_samples,
            )
            num_bo_steps = args.num_batches
            batch_size = args.batch_size

        optimizer_args = {"options": {"gtol" : 1e-9}}

        continuous_optimizer = generate_continuous_optimizer(
            num_initial_samples=args.num_initial_designs,
            num_optimization_runs=args.num_optimization_runs,
            optimizer_args=optimizer_args,
        )

        rule = EfficientGlobalOptimization(
            num_query_points=batch_size,
            builder=acquisition_function,
            optimizer=continuous_optimizer,
        )

    # Create ask-tell optimizer
    optimizer = AskTellOptimizer(
        search_space=search_space,
        datasets=initial_dataset,
        model_specs=model,
        acquisition_rule=rule,
    )

    # Run optimization
    for i in range(num_bo_steps):

        log_regret = save_results(iteration=i, path=path, optimizer=optimizer, minimum=minimum)

        print(f"Batch {i}: Log10 regret {log_regret:.3f}")
        query_batch = optimizer.ask()
        query_values = observer(query_batch)
        optimizer.tell(query_values)


    save_results(iteration=i, path=path, optimizer=optimizer, minimum=minimum)



if __name__ == "__main__":
    main()
