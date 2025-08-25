#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4624
#SBATCH --output=run_YuE_infer%j.out

# find /vol/bitbucket/al4624/inference_cache/transformer_cache/  -type f -delete
# find /vol/bitbucket/al4624/inference_cache/hf_home_cache/  -type f -delete
# find /vol/bitbucket/al4624/inference_cache/xdg_cache_home/  -type f -delete
python infer_multi.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-icl \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --gen_duration 10.0 \
    --use_audio_prompt \
    --audio_prompts_dir_path /homes/al4624/Documents/YuE_finetune/inference_testing_dataset/split_audio_prompts \
    --track_info_json /homes/al4624/Documents/YuE_finetune/inference_testing_dataset/track_info/info.json \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir /homes/al4624/Documents/YuE_finetune/inference_testing_dataset/generated \
    --repetition_penalty 1.1

# python infer_original.py \
#     --cuda_idx 0 \
#     --stage1_model /vol/bitbucket/al4624/model_output \
#     --stage2_model m-a-p/YuE-s2-1B-general \
#     --genre_txt ../prompt_egs/genre.txt \
#     --lyrics_txt ../prompt_egs/lyrics.txt \
#     --use_audio_prompt \
#     --audio_prompt_path redundant_path \
#     --run_n_segments 2 \
#     --stage2_batch_size 4 \
#     --output_dir ../output \
#     --max_new_tokens 3000 \
#     --repetition_penalty 1.1
