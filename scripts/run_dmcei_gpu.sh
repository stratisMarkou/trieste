#!/bin/bash

# Activate virtual environment
. venv-trieste/bin/activate

# Disable GPUs to ensure running on CPUs
export CUDA_VISIBLE_DEVICES=1

# Parameters
objective_dimension=4
objectives=("matern52")
batch_size=10
num_batches=9
num_initial_designs=10000
num_mcei_samples=1000
num_objective_seeds=2
num_search_seeds=39


# =============================================================================
# Decoupled Batch MCEI acquisition
# =============================================================================

for obj in ${objectives[@]}
do
    for objective_seed in $( seq 0 $num_objective_seeds )
    do
        for search_seed in $( seq 0 $num_search_seeds )
        do
            python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                dmcei \
                $obj \
                -batch_size=$batch_size \
                -num_batches=$num_batches \
                --search_seed=$search_seed \
                --objective_seed=$objective_seed \
                --num_initial_designs=$num_initial_designs \
                --num_mcei_samples=$num_mcei_samples \
                --objective_dimension=$objective_dimension
        done
    done
done
