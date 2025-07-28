from pathlib import Path
mixture_track_dir_path_root = "/vol/bitbucket/al4624/finetune_dataset/fma_large/sep/"
mixture_track_paths = [str(file) for file in Path(mixture_track_dir_path_root).rglob('*.mp3') if file.is_file()]
print(mixture_track_paths)