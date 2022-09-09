#!/bin/bash

# Activate virtual environment
. venv-trieste/bin/activate

# Disable GPUs to ensure running on CPUs
export CUDA_VISIBLE_DEVICES=''

# Parameters
objective_dims=(4)
batch_size=1
num_batches=1
num_initial_designs=1000
num_mcei_samples=(10 100 1000)
beta=0.1

# Run exact EI experiments
for search_seed in {0..9}
do
  for objective_seed in {0..2}
  do
    for dim in ${objective_dims[@]}
    do
      # Run exact EI acquisition experiment
      python /scratches/cblgpu07/em626/trieste/scripts/run.py \
             ei \
             matern52 \
             -search_seed=$search_seed \
             -batch_size=$batch_size \
             -num_batches=$num_batches \
             --num_initial_designs=$num_initial_designs \
             --objective_dimension=$dim &
    done
  done
done
wait

# Run batch BO experiments
for search_seed in {0..9}
do
  for objective_seed in {0..2}
  do
    for dim in ${objective_dims[@]}
    do
      # Run random acquisition experiment
      python /scratches/cblgpu07/em626/trieste/scripts/run.py \
             random \
             matern52 \
             -search_seed=$search_seed \
             -batch_size=$batch_size \
             -num_batches=$num_batches \
             --objective_seed=$objective_seed \
             --num_initial_designs=$num_initial_designs \
             --objective_dimension=$dim &

      # Run Thompson acquisition experiment
      python /scratches/cblgpu07/em626/trieste/scripts/run.py \
             thompson \
             matern52 \
             -search_seed=$search_seed \
             -batch_size=$batch_size \
             -num_batches=$num_batches \
             --objective_seed=$objective_seed \
             --num_initial_designs=$num_initial_designs \
             --objective_dimension=$dim &

      for num_samples in ${num_mcei_samples[@]}
      do
        python /scratches/cblgpu07/em626/trieste/scripts/run.py \
          amcei \
          matern52 \
          -search_seed=$search_seed \
          -batch_size=$batch_size \
          -num_batches=$num_batches \
          --objective_seed=$objective_seed \
          --num_mcei_samples=$num_samples \
          --num_initial_designs=$num_initial_designs \
          --beta=$beta \
          --objective_dimension=$dim &
      done

      for num_samples in ${num_mcei_samples[@]}
      do
        python /scratches/cblgpu07/em626/trieste/scripts/run.py \
          mcei \
          matern52 \
          -search_seed=$search_seed \
          -batch_size=$batch_size \
          -num_batches=$num_batches \
          --objective_seed=$objective_seed \
          --num_mcei_samples=$num_samples \
          --num_initial_designs=$num_initial_designs \
          --objective_dimension=$dim &
      done
    done
  done
  wait
done