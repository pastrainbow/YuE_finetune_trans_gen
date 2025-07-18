#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=test_xcodec_infer%j.out
NUM_QUANTIZERS=$1
python xcodec_infer.py --num_quantizers $NUM_QUANTIZERS
