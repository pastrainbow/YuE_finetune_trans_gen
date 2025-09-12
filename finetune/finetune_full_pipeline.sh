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
    # take the last line of output and extract just the number
    jobid=$(ssh -t -q $REMOTE_HOST "bash -i -c '$FINETUNE_JOB_COMMAND'" | tail -n 1 | grep -o '[0-9]\+')
    echo "Submitted finetune job with ID: '$jobid'. Waiting for job to finish..."

    # wait for job to finish
    while true; do
        #wait for the job to be recognised by the scheduler first
        sleep 5
        state=$(ssh -q $REMOTE_HOST "scontrol show job $jobid" | awk -F= '/JobState=/ {print $2}' | awk '{print $1}')
        if [[ "$state" != "RUNNING" && "$state" != "PENDING" ]]; then
            break
        fi
    done

    if [[ "$state" == "COMPLETED" ]]; then
        echo "Finished noise level $NOISE_LEVEL finetuning. Job ID: $jobid"
    else
        echo "Job $jobid failed with state: $state"
        echo "Exiting finetune pipeline."
        exit 1
    fi

    #prepare for next round of finetuning
    mv $MODEL_OUTPUT_DIR $MODEL_NOISE_DIR_PREFIX$NOISE_LEVEL
    mkdir $MODEL_OUTPUT_DIR
    MODEL_NAME=$MODEL_NOISE_DIR_PREFIX$NOISE_LEVEL
done

echo "finetune pipeline completed. Final model at $MODEL_NOISE_DIR_PREFIX${NOISE_LEVELS[-1]}"







