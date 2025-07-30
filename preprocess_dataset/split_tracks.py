import os
from pathlib import Path
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from functools import partial

track_path = "/homes/al4624/Documents/YuE_finetune/finetune_testing_dataset/mixture_audio"
seg_split_dir_path = "/homes/al4624/Documents/YuE_finetune/finetune_testing_dataset"
file_names = [str(file) for file in Path(track_path).glob('*.mp3') if file.is_file()]

def split_file(file_name):
    try:
        #each sample value is from -1 to 1
        audio_data, sample_rate = sf.read(file_name)
        frame_count = len(audio_data)
        #split to 3 segments: start, middle and end
        segment_frame_count = int(frame_count / 3)
        beginning_segment_end = segment_frame_count
        end_segment_start = segment_frame_count * 2
        audio_data_beginning = audio_data[ : beginning_segment_end]
        audio_data_end = audio_data[end_segment_start : ]
        
        sf.write(os.path.join(seg_split_dir_path, Path(file_name).stem + ".beginning.mp3"), audio_data_beginning, sample_rate)

        sf.write(os.path.join(seg_split_dir_path, Path(file_name).stem + ".end.mp3"), audio_data_end, sample_rate)
        

        print(f"File {file_name} finished splitting.")
    #FMA dataset has corrupted files. It is normal for a few files to fail the processing.
    except Exception as e:
        print(f"Error processing {file_name}: {e}. Skipping")
        raise


def parallel_splitting():
    if __name__ == "__main__":
        #ProcessPoolExecutor is probably faster, but we have file IO with soundfile, which will cause problem
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.map(split_file, file_names)
            ]

parallel_splitting()