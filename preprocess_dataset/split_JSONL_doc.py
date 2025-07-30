jsonl_doc_path = "/homes/al4624/Documents/YuE_finetune/YuE_finetune_trans_gen/finetune/example/jsonl/trans_gen.msa.xcodec_16k.jsonl"
# Read the lines from the file
with open(jsonl_doc_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Split into two halves
half = len(lines) // 2
first_half = lines[:half]
second_half = lines[half:]

with open("first_half.jsonl", "w", encoding="utf-8") as f:
    f.writelines(first_half)

with open("second_half.jsonl", "w", encoding="utf-8") as f:
    f.writelines(second_half)