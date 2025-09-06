#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=xcodec_infer_noise_1.0%j.out
NUM_QUANTIZERS=8
python xcodec_infer.py  --num_quantizers $NUM_QUANTIZERS \
                        --inst \
                        --vocals \
                        --mixture \
                        --noised_inst \
                        --noise_levels 1.0 \
