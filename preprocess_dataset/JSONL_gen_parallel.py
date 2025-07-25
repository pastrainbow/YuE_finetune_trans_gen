import pandas as pd
import ast
import json
import os
import soundfile as sf
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


# -------- Load track and genre data once --------
track_df_path = "/homes/al4624/Documents/YuE_finetune/YuE_finetune_trans_gen/preprocess_dataset/fma_metadata/tracks.csv"
genre_df_path = "/homes/al4624/Documents/YuE_finetune/YuE_finetune_trans_gen/preprocess_dataset/fma_metadata/genres.csv"
track_df = pd.read_csv(track_df_path, skiprows=1, dtype={'Unnamed: 0': str}).drop(index=0)
track_df = track_df.rename(columns={'Unnamed: 0': 'track_id'})[['track_id', 'genre_top', 'genres', 'genres_all']]
track_df['track_id'] = track_df['track_id'].astype(int)
genre_df = pd.read_csv(genre_df_path)


parser = argparse.ArgumentParser()
# Model Configuration:
parser.add_argument("--audio_dir_path", type=str, help="Audio mixture directory for the tracks that will be included")
args = parser.parse_args()
audio_dir_path = args.audio_dir_path


codes_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large_sep_codes"
noised_inst_codes_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large_inst_noised_codes"
output_jsonl_path = "/vol/bitbucket/al4624/finetune_dataset/example/jsonl/trans_gen.msa.xcodec_16k.jsonl"

start_lyrics = "[beginning]\n\n"
middle_lyrics = "[middle]\n\n"
end_lyrics = "[end]\n\n"

codec_fps = 50

def get_genres_from_id(track_id):
    row = track_df[track_df['track_id'] == track_id]
    genre_ids = ast.literal_eval(row['genres_all'].values[0])
    genre_strs = []
    for genre_id in genre_ids:
        genre_row = genre_df[genre_df['genre_id'] == genre_id]
        genre_strs.append(genre_row['title'].values[0])
    return genre_strs

def process_track(track_info):
    current_id, track_name = track_info
    json_obj = {}
    json_obj["id"] = str(current_id + 1)

    mixture_audio_path = os.path.join(audio_dir_path, track_name + ".mp3")
    vocals_codes_path = os.path.join(codes_dir_path, track_name + ".Vocals.npy")
    instrumental_codes_path = os.path.join(codes_dir_path, track_name + ".Instrumental.npy")
    noised_inst_codes_path = os.path.join(noised_inst_codes_dir_path, track_name + ".Instrumental.noised.npy")

    for path, desc in [
        (vocals_codes_path, "vocal codec"),
        (instrumental_codes_path, "instrumental codec"),
        (noised_inst_codes_path, "noised instrumental codec"),
        (mixture_audio_path, "mixture audio")
    ]:
        if not os.path.exists(path):
            print(f"Missing {desc} file for track {track_name}!")
            return None

    try:
        track_info_obj = sf.info(mixture_audio_path)
        if track_info_obj.frames == 0 or track_info_obj.samplerate == 0:
            print(f"Invalid audio metadata for file {track_name}. Skipping")
            return None
    except Exception as e:
        print(f"Error processing file {track_name}: {e}. Skipping")
        return None

    track_duration = round(track_info_obj.frames / track_info_obj.samplerate, 2)
    segment_duration = round(track_duration / 3, 2)
    segment_codes_duration = int(segment_duration * codec_fps)

    json_obj["codec"] = "" # Unused
    json_obj["vocals_codec"] = vocals_codes_path
    json_obj["instrumental_codec"] = instrumental_codes_path
    json_obj["noised_instrumental_codec"] = noised_inst_codes_path
    json_obj["audio_length_in_sec"] = track_duration
    json_obj["msa"] = [
        {"start": 0.0, "end": segment_duration, "label": "beginning"},
        {"start": segment_duration, "end": segment_duration * 2.0, "label": "middle"},
        {"start": segment_duration * 2.0, "end": track_duration, "label": "end"}
    ]

    try:
        json_obj["genres"] = ' '.join(get_genres_from_id(int(track_name)))
    except Exception as e:
        print(f"Genre lookup failed for {track_name}: {e}")
        return None

    json_obj["splitted_lyrics"] = {
        "segmented_lyrics": [
            {"offset": 0.0, "duration": segment_duration, "codec_frame_start": 0,
             "codec_frame_end": segment_codes_duration, "line_content": start_lyrics},
            {"offset": segment_duration, "duration": segment_duration, "codec_frame_start": segment_codes_duration,
             "codec_frame_end": segment_codes_duration * 2, "line_content": middle_lyrics},
            {"offset": segment_duration * 2, "duration": segment_duration,
             "codec_frame_start": segment_codes_duration * 2,
             "codec_frame_end": int(track_duration * codec_fps), "line_content": end_lyrics}
        ]
    }

    return json.dumps(json_obj, separators=(",", ":")) + "\n"

# -------- Parallel processing --------

if __name__ == "__main__":
    print("begin JSONL generation")

    track_names = list(set([file.stem for file in Path(audio_dir_path).glob('*.mp3') if file.is_file()]))
    track_infos = [(i, track_name) for i, track_name in enumerate(track_names)]

    jsonl_lines = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_track, info) for info in track_infos]
        for future in as_completed(futures):
            result = future.result()
            if result:
                jsonl_lines.append(result)

    with open(output_jsonl_path, "w") as f:
        f.writelines(jsonl_lines)

    print("Finished JSONL file generation")