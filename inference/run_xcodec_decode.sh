#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=test_xcodec_decode%j.out
python xcodec_decode.py
