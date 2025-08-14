#!/bin/bash
#email results, store console logs in a .out file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=debug%j.out
python scripts/debug.py
