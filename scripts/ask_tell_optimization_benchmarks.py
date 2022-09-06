import argparse

import gpflow
import numpy as np
import scipy as scp
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.ask_tell_optimization import AskTellOptimizer

from trieste.acquisition.rule import (
    RandomSampling,
    EfficientGlobalOptimization,
)

from trieste.models.gpflow.builders import (
    build_gpr,
)

from trieste.acquisition.function import (
    BatchMonteCarloExpectedImprovement,
    GreedyContinuousThompsonSampling,
)

import trieste
from trieste.objectives.utils import mk_observer
from trieste.objectives.single_objectives import *


objectives = {
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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "objective",
        type=str,
        choices=objectives.keys(),
    )

    parser.add_argument(
        "acquisition",
        type=str,
        choices=[
            "random",
            "thompson",
            "ei",
        ],
    )

    parser.add_argument(
        "-seed",
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
        "--trainable_noise",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--num_ei_samples",
        type=int,
    )

    args = parser.parse_args()

    # Set random seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Set objective
    objective, minimum, search_space = objectives[args.objective]

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

    else:

        if args.acquisition == "thompson":
            acquisition_function = GreedyContinuousThompsonSampling()

        elif args.acquisition == "ei":
            acquisition_function = BatchMonteCarloExpectedImprovement(
                sample_size=args.num_ei_samples,
            )

        rule = EfficientGlobalOptimization(
            num_query_points=args.batch_size,
            builder=acquisition_function,
        )

    # Create ask-tell optimizer
    optimizer = AskTellOptimizer(
        search_space=search_space,
        datasets=initial_dataset,
        model_specs=model,
        acquisition_rule=rule,
    )

    # Run optimization
    for i in range(args.num_batches):

        query_batch = optimizer.ask()
        print(optimizer.model.model.kernel.lengthscales)
        query_values = observer(query_batch)
        optimizer.tell(query_values)

        print(
            log10_regret(
                queries=optimizer.to_result().try_get_final_dataset().observations,
                minimum=minimum,
            )
        )


if __name__ == "__main__":
    main()
