#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=finetune_full_pipeline%j.out
MODEL_OUTPUT_DIR="/vol/bitbucket/al4624/YuE_finetune_ouput/cache_output"
MODEL_NOISE_DIR_PREFIX="/vol/bitbucket/al4624/YuE_finetune_ouput/noise_"
FINETUNE_JOB_COMMAND="source ~/.bashrc && YuE_finetune_init && cd finetune && sbatch --partition AMD7-A100-T --mem=48G --parsable scripts/run_finetune.sh"
REMOTE_HOST="gpu_cluster_2"

NOISE_LEVELS=(0.1 0.3 0.5 0.7 1.0)
MODEL_NAME="m-a-p/YuE-s1-7B-anneal-en-icl"

for NOISE_LEVEL in "${NOISE_LEVELS[@]}"; do
    bash finetune_preprocess.sh 0 $NOISE_LEVEL
    echo "Finished preprocessing dataset for noise level $NOISE_LEVEL."

    python update_base_model.py --model_name $MODEL_NAME
    echo "Finished setting base model to foundation model. Starting noise level $NOISE_LEVEL finetuning."
    
    # submit job remotely and capture job ID
    jobid=$(ssh -t -q $REMOTE_HOST "bash -i -c '$FINETUNE_JOB_COMMAND'" | tail -n 1)
    echo "Submitted finetune job with ID $jobid"

    # wait for job to finish
    # IMPORTANT: can't run sacct command, so this just waits until job is no longer in queue
    # and assumes job completed successfully
    while [[ "$(ssh -q $REMOTE_HOST "squeue -j $jobid")" == *$'\n'* ]]; do
        sleep 5
    done

    echo "Finished noise level $NOISE_LEVEL finetuning. Job ID: $jobid"

    #prepare for next round of finetuning
    mv $MODEL_OUTPUT_DIR $MODEL_NOISE_DIR_PREFIX$NOISE_LEVEL
    mkdir $MODEL_OUTPUT_DIR
    MODEL_NAME=$MODEL_NOISE_DIR_PREFIX$NOISE_LEVEL
done

echo "finetune pipeline completed. Final model at $MODEL_NOISE_DIR_PREFIX${NOISE_LEVELS[-1]}"







