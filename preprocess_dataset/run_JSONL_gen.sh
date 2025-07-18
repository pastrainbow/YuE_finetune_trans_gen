#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=JSONL_gen%j.out
AUDIO_DIR_PATH=$1
python JSONL_gen_parallel.py --audio_dir_path $AUDIO_DIR_PATH
