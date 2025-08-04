#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=calc_KL_div%j.out
python KL_div_calc.py