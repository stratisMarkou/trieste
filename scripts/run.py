import shutil
import sys
sys.path.append("./scripts")

import argparse

import tensorflow as tf
import gpflow
import numpy as np

import trieste
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.acquisition.optimizer import generate_continuous_optimizer

from trieste.acquisition.rule import (
    RandomSampling,
    EfficientGlobalOptimization,
    EfficientGlobalOptimizationWithPreOptimization,
)

from trieste.models.gpflow.builders import (
    build_gpr,
)

from trieste.acquisition.function import (
    ExpectedImprovement,
    BatchMonteCarloExpectedImprovement,
    DecoupledBatchMonteCarloExpectedImprovement,
    AnnealedBatchMonteCarloExpectedImprovement,
    GreedyContinuousThompsonSampling,
)

from trieste.acquisition.utils import split_acquisition_function_calls

from trieste.objectives.utils import mk_observer
from trieste.objectives.single_objectives import *

from utils import make_gp_objective, arg_summary, set_up_logging, gp_objectives


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
    "ackley_5": (
        ackley_5,
        ACKLEY_5_MINIMUM,
        ACKLEY_5_SEARCH_SPACE,
    ),
}



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
            "dmcei",
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
        "-batch_size",
        type=int,
    )

    parser.add_argument(
        "-num_batches",
        type=int,
    )

    parser.add_argument(
        "--search_seed",
        type=int,
        default=0,
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
        default=1.,
    )

    parser.add_argument(
        "--num_gp_rff_features",
        type=int,
        default=64,
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
        "--num_amcei_steps",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num_mcei_samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--linear_stddev",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--gtol",
        type=float,
        default=1e-20,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="experiments",
    )

    # Parse arguments and print argument summary
    args = parser.parse_args()
    summary = arg_summary(args)
    print(summary)

    assert not (args.objective in gp_objectives and args.objective_dimension == -1)

    # Set up logging: save path, argument logging, regret logging, tensorboard
    path = set_up_logging(args=args, summary=summary)

    # Set objective
    if args.objective in standard_objectives:
        objective, minimum, search_space = standard_objectives[args.objective]

    else:

        # Set objective seed
        tf.random.set_seed(args.objective_seed)
        np.random.seed(args.objective_seed)

        # Make objective
        objective, minimum, search_space = make_gp_objective(
            kernel=args.objective,
            dim=args.objective_dimension,
            num_fourier_components=args.objective_fourier_components,
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

    model = GaussianProcessRegression(
        model,
        num_rff_features=args.num_gp_rff_features,
    )

    # Create acquisition rule
    if args.acquisition == "random":
        rule = RandomSampling(num_query_points=args.batch_size)
        num_bo_steps = args.num_batches
        initial_design_split_size = args.num_initial_designs

    else:

        if args.acquisition == "thompson":
            acquisition_function = GreedyContinuousThompsonSampling()
            num_bo_steps = args.num_batches
            batch_size = args.batch_size
            initial_design_split_size = args.num_initial_designs
            pre_acquisition_functions = None

        elif args.acquisition == "ei":
            acquisition_function = ExpectedImprovement()
            num_bo_steps = args.batch_size * args.num_batches
            batch_size = 1
            initial_design_split_size = args.num_initial_designs
            pre_acquisition_functions = None

        elif args.acquisition == "mcei":
            acquisition_function = BatchMonteCarloExpectedImprovement(
                sample_size=args.num_mcei_samples,
            )
            num_bo_steps = args.num_batches
            batch_size = args.batch_size
            initial_design_split_size = args.num_initial_designs
            pre_acquisition_functions = None

        elif args.acquisition == "dmcei":
            acquisition_function = DecoupledBatchMonteCarloExpectedImprovement(
                sample_size=args.num_mcei_samples,
            )
            num_bo_steps = args.num_batches
            batch_size = args.batch_size
            initial_design_split_size = 1000
            pre_acquisition_functions = None

        elif args.acquisition == "amcei":

            pre_acquisition_functions = [
                AnnealedBatchMonteCarloExpectedImprovement(
                    beta=args.beta*2**-i,
                    sample_size=args.num_mcei_samples,
                ) for i in range(args.num_amcei_steps+1)
            ]

            acquisition_function = BatchMonteCarloExpectedImprovement(
                sample_size=args.num_mcei_samples,
            )

            num_bo_steps = args.num_batches
            batch_size = args.batch_size
            initial_design_split_size = args.num_initial_designs

        optimizer_args = {"options": {"gtol" : args.gtol}}

        continuous_optimizer = generate_continuous_optimizer(
            num_initial_samples=args.num_initial_designs,
            num_optimization_runs=args.num_optimization_runs,
            optimizer_args=optimizer_args,
        )

#         continuous_optimizer = split_acquisition_function_calls(
#             continuous_optimizer,
#             split_size=initial_design_split_size,
#         )

        rule = EfficientGlobalOptimizationWithPreOptimization(
            num_query_points=batch_size,
            pre_builders=pre_acquisition_functions,
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

    log_regret = save_results(iteration=i+1, path=path, optimizer=optimizer, minimum=minimum)
    print(f"Batch {i+1}: Log10 regret {log_regret:.3f}")


if __name__ == "__main__":
    main()
