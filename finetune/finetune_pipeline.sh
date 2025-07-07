#clear previous log and binary files
rm -rf count_token_logs
rm -rf example/mmap
#preprocess dataset, then start finetuning
bash scripts/preprocess_data.sh trans_gen cot
bash scripts/preprocess_data.sh trans_gen icl_cot
bash scripts/count_tokens.sh ./example/mmap/
python core/parse_mixture.py -c example/trans_gen_data_mixture_cfg.yml
bash scripts/run_finetune.sh

sbatch --partition AMD7-A100-T ~/Documents/YuE_finetune/YuE_finetune_trans_gen/finetune/scripts/run_finetune.sh
