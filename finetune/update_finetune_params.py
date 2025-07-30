data_path = ""
train_iters = 0
with open("example/mixture_parse_log.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for x in range(len(lines)):
        line = lines[x]
        if line.startswith("[CRITICAL] DATA_PATH"):
            data_path = lines[x+1].replace(" \n", "")
        elif line.startswith("[CRITICAL] TRAIN_ITERS"):
            train_iters = int(float(lines[x+1].replace("\n", "")))
finetune_script_lines = []
with open("scripts/run_finetune.sh", "r", encoding="utf-8") as f:
    finetune_script_lines = f.readlines()

for (line_count, line) in enumerate(finetune_script_lines):
    if line.startswith("DATA_PATH"):
        finetune_script_lines[line_count] = f"DATA_PATH=\"{data_path}\"\n"
        print(f"Modified finetune parameter DATA_PATH, written content: {finetune_script_lines[line_count]}")
    elif line.startswith("TRAIN_ITERS"):
        finetune_script_lines[line_count] = f"TRAIN_ITERS={train_iters}\n"
        print(f"Modified finetune parameter TRAIN_ITERS, written content: {finetune_script_lines[line_count]}")
with open("scripts/run_finetune.sh", "w", encoding="utf-8") as f:
    f.writelines(finetune_script_lines)