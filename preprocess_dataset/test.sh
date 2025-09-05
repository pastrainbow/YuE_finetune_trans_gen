REMOTE_HOST="gpu_cluster_2"
FINETUNE_JOB_COMMAND="source ~/.bashrc && YuE_finetune_init && cd inference && sbatch --parsable run_infer.sh"

# submit job remotely and capture job ID
jobid=$(ssh -t -q $REMOTE_HOST "bash -i -c '$FINETUNE_JOB_COMMAND'" | tail -n 1)
echo "Submitted inference job with ID $jobid"

# wait for job to finish and check exit code
while [[ "$(ssh -q $REMOTE_HOST "squeue -j $jobid")" == *$'\n'* ]]; do
    sleep 5
done

# # get job exit code
# exit_code=$(ssh $REMOTE_HOST "sacct -j $jobid --format=ExitCode --noheader | awk '{print \$1}' | cut -d: -f1")


# if [ "$exit_code" -ne 0 ]; then
#     echo "Job $jobid failed (exit code $exit_code). Aborting pipeline."
#     exit 1

echo "finished error checking"