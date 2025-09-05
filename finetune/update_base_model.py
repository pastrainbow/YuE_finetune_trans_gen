import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Name or path of the base model")
args = parser.parse_args()

finetune_script_lines = []
with open("scripts/run_finetune.sh", "r", encoding="utf-8") as f:
    finetune_script_lines = f.readlines()

for (line_count, line) in enumerate(finetune_script_lines):
    if line.startswith("MODEL_NAME="):
        finetune_script_lines[line_count] = f"MODEL_NAME=\"{args.model_name}\"\n"
        print(f"Modified finetune parameter MODEL_NAME, written content: {finetune_script_lines[line_count]}")
with open("scripts/run_finetune.sh", "w", encoding="utf-8") as f:
    f.writelines(finetune_script_lines)