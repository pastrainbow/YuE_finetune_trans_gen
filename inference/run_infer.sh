#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=run_YuE_infer%j.out
#10 second transition
python infer.py \
    --cuda_idx 0 \
    --stage1_model /vol/bitbucket/al4624/model_output \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --max_new_tokens 500 \
    --gen_duration 10.0 \
    --use_audio_prompt \
    --start_audio_prompt_path /homes/al4624/Documents/YuE_finetune/test_sep_original/2.mp3 \
    --end_audio_prompt_path /homes/al4624/Documents/YuE_finetune/test_sep_original/10.mp3 \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/template_lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir /vol/bitbucket/al4624/inference_output \
    --max_new_tokens 3000 \