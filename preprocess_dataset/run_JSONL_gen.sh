#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=JSONL_gen%j.out
python JSONL_gen_parallel.py
