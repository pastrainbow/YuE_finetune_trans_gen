import os
from pathlib import Path
input_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large/"
output_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large_sep/"
input_track_paths = [str(file) for file in Path(input_dir_path).rglob('*.mp3') if file.is_file()]
output_track_paths =[str(file) for file in Path(output_dir_path).rglob('*.mp3') if file.is_file()]
file_count = 0
for input_track_path in input_track_paths:
    name = os.path.basename(input_track_path)
    vocals_path = os.path.join(output_dir_path, name, ".Vocals.mp3")
    instr_path = os.path.join(output_dir_path, name, ".Instrumental.mp3")
    if (not (os.path.exists(vocals_path) and os.path.exists(instr_path))):
        print(f"File {name} did not get separated.")
    else:
        file_count += 1
print(f"{file_count} files got separated.")
