#clear previous log
rm -rf count_token_logs

#clear cache
rm -rf /vol/bitbucket/al4624/finetune_cache/data_cache/*
rm -rf /vol/bitbucket/al4624/finetune_cache/model_cache/*

#preprocess dataset, then start finetuning
bash scripts/preprocess_data.sh trans_gen icl_cot inst
bash scripts/count_tokens.sh ./example/mmap/
python core/parse_mixture.py -c example/trans_gen_data_mixture_cfg.yml > example/mixture_parse_log.txt
python update_finetune_params.py

# sbatch --partition AMD7-A100-T scripts/run_finetune.sh
