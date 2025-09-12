#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=test%j.out

#must have the parsable flag to get job ID
FINETUNE_JOB_COMMAND="source ~/.bashrc && ace_step_init && sbatch --parsable --partition training infer.sh"
REMOTE_HOST="gpu_cluster_2"

# submit job remotely and capture job ID
# take the last line of output and extract just the number
jobid=$(ssh -t -q $REMOTE_HOST "bash -i -c '$FINETUNE_JOB_COMMAND'" | tail -n 1 | grep -o '[0-9]\+')
echo "Submitted finetune job with ID: '$jobid'. Waiting for job to finish..."

# wait for job to finish
while true; do
    #wait for the job to be recognised by the scheduler first
    sleep 5s
    # echo "Checking job status for job ID $jobid..."
    state=$(ssh -q $REMOTE_HOST "scontrol show job $jobid" | awk -F= '/JobState=/ {print $2}' | awk '{print $1}')
    if [[ "$state" != "RUNNING" && "$state" != "PENDING" ]]; then
        break
    fi
    # echo $state
done

if [[ "$state" == "COMPLETED" ]]; then
    echo "Job $jobid completed successfully."
else
    echo "Job $jobid failed with state: $state"
    exit 1
fi















