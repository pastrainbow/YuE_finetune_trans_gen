import pandas as pd
import ast
def get_track_df(track_df_path):
    track_df = pd.read_csv(track_df_path, skiprows=1, dtype={'Unnamed: 0': str})
    track_df = track_df.drop(index=0)
    track_df = track_df.rename(columns={'Unnamed: 0': 'track_id'})
    track_df = track_df[['track_id', 'genre_top', 'genres', 'genres_all']]
    track_df['track_id'] = track_df['track_id'].astype(int)
    return track_df
    
def get_genre_df(genre_df_path):
    return pd.read_csv(genre_df_path)

def get_genres_from_id(track_id, track_df_path, genre_df_path):
    track_df = get_track_df(track_df_path)
    genre_df = get_genre_df(genre_df_path)
    
    row = track_df[track_df['track_id'] == track_id]
    genre_ids = ast.literal_eval(row['genres_all'].values[0])
    genre_strs = []
    for genre_id in genre_ids:
        genre_row = genre_df[genre_df['genre_id'] == genre_id]
        genre_str = genre_row['title'].values[0]
        #print(genre_str)
        genre_strs.append(genre_str)
    return genre_strs

import json
import os
import soundfile as sf
import jsonlines
from pathlib import Path

track_df_path = "/homes/al4624/Documents/YuE_finetune/YuE_finetune_trans_gen/preprocess_dataset/fma_metadata/tracks.csv"
genre_df_path = "/homes/al4624/Documents/YuE_finetune/YuE_finetune_trans_gen/preprocess_dataset/fma_metadata/genres.csv"
start_lyrics = "[beginning]\n\n"
middle_lyrics = "[middle]\n\n"
end_lyrics = "[end]\n\n"
output_jsonl_path = "/homes/al4624/Documents/YuE_finetune/YuE_finetune_trans_gen/finetune/example/jsonl/trans_gen.msa.xcodec_16k.jsonl"
audio_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large/sep/noise_0.1"
codes_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large_sep_codes"
noised_inst_codes_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large_inst_noised_codes"
json_obj = {}
#since bumch of audio files in FMA_large are corrupted, we cannot use the original dataset to get all the track names.
track_names = list(set([file.stem for file in Path(audio_dir_path).glob('*.mp3') if file.is_file()]))

jsonl_string = ""

print("begin JSONL generation")

for current_id, track_name in enumerate(track_names):
    #in the example, id starts at 1
    json_obj["id"] = str(current_id + 1)

    #print(f"Current id: {current_id + 1}")
    
    #only for getting the duration of track
    mixture_audio_path = os.path.join(audio_dir_path, track_name + ".mp3")
    
    vocals_codes_path = os.path.join(codes_dir_path, track_name + ".Vocals.npy")
    instrumental_codes_path = os.path.join(codes_dir_path, track_name + ".Instrumental.npy")
    noised_inst_codes_path = os.path.join(noised_inst_codes_dir_path, track_name + ".Instrumental.noised.npy")

    if (not (os.path.exists(vocals_codes_path))):
        print(f"Missing vocal codec file for track {track_name}!")
        continue

    if (not (os.path.exists(instrumental_codes_path))):
        print(f"Missing instrumental codec file for track {track_name}!")
        continue

    if (not (os.path.exists(mixture_audio_path))):
        print(f"Missing mixture audio file for track {track_name}!")
        continue

    if (not (os.path.exists(noised_inst_codes_path))):
        print(f"Missing noised instrumental codec file for track {track_name}!")
        continue
    
    json_obj["codec"] = "" #Unused
    json_obj["vocals_codec"] = vocals_codes_path
    json_obj["instrumental_codec"] = instrumental_codes_path
    json_obj["noised_instrumental_codec"] = noised_inst_codes_path
    
    #get track duration in seconds, so that we know the split time for start, middle and end
    track_info = None
    try:
        track_info = sf.info(mixture_audio_path)
    except Exception as e:
        print(f"Error processing file {track_name}: {e}. Skipping")
        continue
        
    track_duration = round(track_info.frames / track_info.samplerate, 2)
    
    codec_fps = 50
    segment_duration = round(track_duration / 3, 2)
    segment_codes_duration = int(segment_duration * codec_fps)
    
    msa_start = {"start": 0.0, "end": segment_duration, "label": "beginning"}
    
    msa_middle = {"start": segment_duration, "end": segment_duration * 2.0, "label": "middle"}
    
    msa_end = {"start": segment_duration * 2.0, "end": track_duration, "label": "end"}
    
    json_obj["audio_length_in_sec"] = round(track_duration, 2)
    
    json_obj["msa"] = [msa_start, msa_middle, msa_end]

        
    json_obj["genres"] = ', '.join(get_genres_from_id(int(track_name), track_df_path, genre_df_path))
    
    start_lyric_segment = {"offset": 0.0, 
                           "duration": segment_duration, 
                           "codec_frame_start":0, 
                           "codec_frame_end": segment_codes_duration,
                           "line_content": start_lyrics}
    
    middle_lyric_segment = {"offset": segment_duration, 
                            "duration": segment_duration,
                            "codec_frame_start": segment_codes_duration,
                            "codec_frame_end": segment_codes_duration * 2,
                            "line_content": middle_lyrics}
    
    end_lyric_segment = {"offset": segment_duration * 2,
                         "duration": segment_duration,
                         "codec_frame_start": segment_codes_duration * 2,
                         "codec_frame_end": int(track_duration * codec_fps),
                         "line_content": end_lyrics}
    
    segmented_lyrics =  {"segmented_lyrics": [start_lyric_segment, middle_lyric_segment, end_lyric_segment]}
    
    json_obj["splitted_lyrics"] = segmented_lyrics

    jsonl_string += json.dumps(json_obj, separators=(",", ":")) + "\n"

    #print(f"Current string: {jsonl_string}")
                              
with open(output_jsonl_path, "w") as f:
    f.write(jsonl_string)

print("Finished JSONL file generation")