#clear previous log
rm -rf count_token_logs/*

#clear all cache
rm -rf /vol/bitbucket/al4624/cache/finetune_cache/data_cache/*
rm -rf /vol/bitbucket/al4624/cache/finetune_cache/model_cache/*
rm -rf /vol/bitbucket/al4624/cache/finetune_cache/hf_home_cache/*
rm -rf /vol/bitbucket/al4624/cache/finetune_cache/xdg_cache_home/*

NUM_EXAMPLES=$1
NOISE_LEVEL=$2

#Extract a certain number of examples from the dataset
python split_JSONL_doc.py --num_examples $NUM_EXAMPLES --input_file full_$NOISE_LEVEL.jsonl

#preprocess dataset, update finetune script with correct parameters, then start finetuning
bash scripts/preprocess_data.sh trans_gen icl_cot inst
sleep 20
bash scripts/count_tokens.sh ./example/mmap/
sleep 20
python core/parse_mixture.py -c example/trans_gen_data_mixture_cfg.yml > example/mixture_parse_log.txt
sleep 20
python update_finetune_params.py

# sbatch --partition AMD7-A100-T scripts/run_finetune.sh
