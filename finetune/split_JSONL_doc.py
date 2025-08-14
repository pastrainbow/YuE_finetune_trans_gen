# jsonl_doc_path = "/homes/al4624/Documents/YuE_finetune/YuE_finetune_trans_gen/finetune/example/jsonl/trans_gen.msa.xcodec_16k.jsonl"
# # Read the lines from the file
# with open(jsonl_doc_path, "r", encoding="utf-8") as f:
#     lines = f.readlines()

# # Split into two halves
# half = len(lines) // 2
# first_half = lines[:half]
# second_half = lines[half:]

# with open("first_half.jsonl", "w", encoding="utf-8") as f:
#     f.writelines(first_half)

# with open("second_half.jsonl", "w", encoding="utf-8") as f:
#     f.writelines(second_half)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_examples", type=int, help="Number of examples to include in the JSONL file")
args = parser.parse_args()

jsonl_doc_path = "/homes/al4624/Documents/YuE_finetune/YuE_finetune_trans_gen/finetune/example/jsonl/full.jsonl"
# Read the lines from the file
with open(jsonl_doc_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

line_count = args.num_examples
segment = lines[:line_count]

with open("/homes/al4624/Documents/YuE_finetune/YuE_finetune_trans_gen/finetune/example/jsonl/trans_gen.msa.xcodec_16k.jsonl", "w", encoding="utf-8") as f:
    f.writelines(segment)