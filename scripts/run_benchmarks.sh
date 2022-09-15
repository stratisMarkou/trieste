#!/bin/bash

# Activate virtual environment
. venv-trieste/bin/activate

# Disable GPUs to ensure running on CPUs
export CUDA_VISIBLE_DEVICES=1

# Parameters
objective_dimension=4
objectives=("hartmann_6")
batch_size=5
num_batches=10
num_initial_designs=10000
num_mcei_samples=1000
num_amcei_halvings=5
beta=0.5
num_objective_seeds=0
num_search_seeds=0
max_jobs=20
counter=0


## =============================================================================
## Random acquisition
## =============================================================================
#
#for obj in ${objectives[@]}
#do
#    for objective_seed in $( seq 0 $num_objective_seeds )
#    do
#        for search_seed in $( seq 0 $num_search_seeds )
#        do
#            # Run random acquisition experiment
#            python /scratches/cblgpu07/em626/trieste/scripts/run.py \
#                random \
#                $obj \
#                -batch_size=$batch_size \
#                -num_batches=$num_batches \
#                --search_seed=$search_seed \
#                --objective_seed=$objective_seed \
#                --num_initial_designs=$num_initial_designs \
#                --objective_dimension=$objective_dimension
#
##            if [[ $counter -ge $max_jobs ]]
##            then
##                wait
##                counter=1
##            else
##                counter=$(($counter+1))
##            fi
#        done
##        wait
#    done
#done
#
#
## =============================================================================
## Single-point EI acquisition
## =============================================================================
#
#for obj in ${objectives[@]}
#do
#    for objective_seed in $( seq 0 $num_objective_seeds )
#    do
#        for search_seed in $( seq 0 $num_search_seeds )
#        do
#            python /scratches/cblgpu07/em626/trieste/scripts/run.py \
#              ei \
#              $obj \
#              -batch_size=$batch_size \
#              -num_batches=$num_batches \
#              --search_seed=$search_seed \
#              --objective_seed=$objective_seed \
#              --num_initial_designs=$num_initial_designs \
#              --objective_dimension=$objective_dimension
#        done
#    done
#done
#
#
## =============================================================================
## Single-point MCEI acquisition
## =============================================================================
#
#for obj in ${objectives[@]}
#do
#    for objective_seed in $( seq 0 $num_objective_seeds )
#    do
#        for search_seed in $( seq 0 $num_search_seeds )
#        do
#            python /scratches/cblgpu07/em626/trieste/scripts/run.py \
#              mcei \
#              $obj \
#              -batch_size=1 \
#              -num_batches=$(($batch_size*$num_batches)) \
#              --objective_seed=$objective_seed \
#              --search_seed=$search_seed \
#              --num_mcei_samples=$num_mcei_samples \
#              --num_initial_designs=$num_initial_designs \
#              --objective_dimension=$objective_dimension
#
##            if [[ $counter -ge $max_jobs ]]
##            then
##                wait
##                counter=1
##            else
##                counter=$(($counter+1))
##            fi
#        done
##        wait
#    done
#done


## =============================================================================
## Batch MCEI acquisition
## =============================================================================
#
#for obj in ${objectives[@]}
#do
#    for objective_seed in $( seq 0 $num_objective_seeds )
#    do
#        for search_seed in $( seq 0 $num_search_seeds )
#        do
#            python /scratches/cblgpu07/em626/trieste/scripts/run.py \
#                mcei \
#                $obj \
#                -batch_size=$batch_size \
#                -num_batches=$num_batches \
#                --search_seed=$search_seed \
#                --objective_seed=$objective_seed \
#                --num_initial_designs=$num_initial_designs \
#                --num_mcei_samples=$num_mcei_samples \
#                --objective_dimension=$objective_dimension
#
##            if [[ $counter -ge $max_jobs ]]
##            then
##                wait
##                counter=1
##            else
##                counter=$(($counter+1))
##            fi
#        done
##        wait
#    done
#done

# =============================================================================
# Annealed Batch MCEI acquisition
# =============================================================================

for obj in ${objectives[@]}
do
    for objective_seed in $( seq 0 $num_objective_seeds )
    do
        for search_seed in $( seq 0 $num_search_seeds )
        do
            python /scratches/cblgpu07/em626/trieste/scripts/run.py \
                amcei \
                $obj \
                -batch_size=$batch_size \
                -num_batches=$num_batches \
                --search_seed=$search_seed \
                --objective_seed=$objective_seed \
                --num_initial_designs=$num_initial_designs \
                --num_mcei_samples=$num_mcei_samples \
                --beta=$beta \
                --num_amcei_halvings=$num_amcei_halvings \
                --objective_dimension=$objective_dimension

#            if [[ $counter -ge $max_jobs ]]
#            then
#                wait
#                counter=1
#            else
#                counter=$(($counter+1))
#            fi
        done
#        wait
    done
done


## =============================================================================
## Decoupled Batch MCEI acquisition
## =============================================================================
#
#for obj in ${objectives[@]}
#do
#    for objective_seed in $( seq 0 $num_objective_seeds )
#    do
#        for search_seed in $( seq 0 $num_search_seeds )
#        do
#            python /scratches/cblgpu07/em626/trieste/scripts/run.py \
#                dmcei \
#                $obj \
#                -batch_size=$batch_size \
#                -num_batches=$num_batches \
#                --search_seed=$search_seed \
#                --objective_seed=$objective_seed \
#                --num_initial_designs=$num_initial_designs \
#                --num_mcei_samples=$num_mcei_samples \
#                --objective_dimension=$objective_dimension
#        done
#    done
#done
#
#
## =============================================================================
## Batch Thompson sampling acquisition
## =============================================================================
#
#for obj in ${objectives[@]}
#do
#    for objective_seed in $( seq 0 $num_objective_seeds )
#    do
#        for search_seed in $( seq 0 $num_search_seeds )
#        do
#            # Run random acquisition experiment
#            python /scratches/cblgpu07/em626/trieste/scripts/run.py \
#                thompson \
#                $obj \
#                -batch_size=$batch_size \
#                -num_batches=$num_batches \
#                --search_seed=$search_seed \
#                --objective_seed=$objective_seed \
#                --num_initial_designs=$num_initial_designs \
#                --objective_dimension=$objective_dimension
#
##            if [[ $counter -ge $max_jobs ]]
##            then
##                wait
##                counter=1
##            else
##                counter=$(($counter+1))
##            fi
#        done
##        wait
#    done
#done
