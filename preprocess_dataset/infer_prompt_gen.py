import os
import pandas as pd
from pathlib import Path
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import json
import ast

# track_path = "/homes/al4624/Documents/YuE_finetune/inference_testing_dataset/full_audio"
# seg_split_dir_path = "/homes/al4624/Documents/YuE_finetune/inference_testing_dataset/split_audio_prompts"
# track_info_json_path = "/homes/al4624/Documents/YuE_finetune/inference_testing_dataset/track_info/info.json" 
# file_paths = [str(file) for file in Path(track_path).glob('*.mp3') if file.is_file()]

track_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large/sep/eval"
inst_track_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large_sep"
seg_split_dir_path = "/vol/bitbucket/al4624/eval_dataset/split_segments"
track_info_json_path = "/vol/bitbucket/al4624/eval_dataset/info/info.json" 
file_paths = [str(file) for file in Path(track_path).rglob('*.mp3') if file.is_file()]


# -------- Load track and genre data once --------
track_df_path = "/homes/al4624/Documents/YuE_finetune/YuE_finetune_trans_gen/preprocess_dataset/fma_metadata/tracks.csv"
genre_df_path = "/homes/al4624/Documents/YuE_finetune/YuE_finetune_trans_gen/preprocess_dataset/fma_metadata/genres.csv"
track_df = pd.read_csv(track_df_path, skiprows=1, dtype={'Unnamed: 0': str}).drop(index=0)
track_df = track_df.rename(columns={'Unnamed: 0': 'track_id'})[['track_id', 'genre_top', 'genres', 'genres_all']]
track_df['track_id'] = track_df['track_id'].astype(int)
genre_df = pd.read_csv(genre_df_path)


def get_genres_from_id(track_id):
    row = track_df[track_df['track_id'] == track_id]
    genre_ids = ast.literal_eval(row['genres_all'].values[0])
    genre_strs = []
    for genre_id in genre_ids:
        genre_row = genre_df[genre_df['genre_id'] == genre_id]
        genre_strs.append(genre_row['title'].values[0])
    return genre_strs


def split_file(file_path):
    try:
        track_name = Path(file_path).stem
        
        info_json = {}
        
        inst_track_path = os.path.join(inst_track_dir_path, track_name + '.Instrumental.mp3')

        if not os.path.exists(inst_track_path):
            print(f"[ERROR] Instrumental track for track {track_name} does not exist! Skipping")
            return None
        

        #each sample value is from -1 to 1
        audio_data, sample_rate = sf.read(inst_track_path)
        frame_count = len(audio_data)
        #split to 3 segments: start, middle and end
        segment_frame_count = int(frame_count / 3)
        beginning_segment_end = segment_frame_count
        end_segment_start = segment_frame_count * 2
        audio_data_beginning = audio_data[ : beginning_segment_end]
        audio_data_middle = audio_data[beginning_segment_end : end_segment_start]
        audio_data_end = audio_data[end_segment_start : ]

        info_json["track_name"] = track_name
        info_json["genres"] = get_genres_from_id(int(track_name))
        info_json["endpoint_1"] = beginning_segment_end / sample_rate
        info_json["endpoint_2"] = end_segment_start / sample_rate
        
        sf.write(os.path.join(seg_split_dir_path, track_name + ".Instrumental.beginning.mp3"), audio_data_beginning, sample_rate)
        sf.write(os.path.join(seg_split_dir_path, track_name + ".Instrumental.middle.mp3"), audio_data_middle, sample_rate)
        sf.write(os.path.join(seg_split_dir_path, track_name + ".Instrumental.end.mp3"), audio_data_end, sample_rate)

        print(f"Track {track_name} finished splitting.")

        return info_json
    #FMA dataset has corrupted files. It is normal for a few files to fail the processing.
    except Exception as e:
        print(f"Error processing {file_path}: {e}. Skipping")
        return None


def parallel_splitting():
    if __name__ == "__main__":
        track_infos = []
        
        #ProcessPoolExecutor is probably faster, but we have file IO with soundfile, which will cause problem
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(split_file, file_path) for file_path in file_paths]

            for future in as_completed(futures):
                info = future.result()
                if info:
                    track_infos.append(info)
        json_obj = {"infos": track_infos}
        with open(track_info_json_path, 'w') as json_file:
                json.dump(json_obj, json_file, indent=4)
        print("Successfully generated split endpoint info json file")
            
parallel_splitting()