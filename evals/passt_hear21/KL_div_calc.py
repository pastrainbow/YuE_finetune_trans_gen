import os
os.environ['HF_HOME'] = '/vol/bitbucket/al4624/cache/general_cache/hf_home_cache'
os.environ['XDG_CACHE_HOME'] = '/vol/bitbucket/al4624/cache/general_cache/xdg_cache_home'
import torch
torch.multiprocessing.set_start_method("spawn", force=True)
import torchaudio
from torchaudio.transforms import Resample
import torch.nn.functional as F
from hear21passt.base30sec import load_model
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

torchaudio.set_audio_backend("ffmpeg")

def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio


def calc_tags_prob(model, wave_signal):
    wave_signal = wave_signal.cuda()
    logits = model(wave_signal.cuda())
    #pytorch's KL divergence function accepts log softmax activat    # print(f"sum: {torch.sum(probs)}")ed probabilities
    probs = F.log_softmax(logits, dim=1)
    # print(f"sum: {torch.sum(probs)}")
    return probs


def calc_KL_div(prob_ground_truth, prob_gen):
    return F.kl_div(prob_ground_truth, prob_gen, reduction='batchmean', log_target=True)

def calc_KL_div_for_track(model, track_path, track_info_file_path):
    track_name = Path(track_path).stem
    track_infos = None
    with open(track_info_file_path) as f:
        track_infos = json.load(f)['infos']
    track_info = None
    for info in track_infos:
        if info["track_name"] == track_name:
            track_info = info
            break
    if track_info is None:
        raise LookupError(f"Info for track {track_name} not found!")
    sample_rate = 16000
    #endpoints are in seconds, so that it is independent of the sample rate
    endpoint_1 = track_info['endpoint_1']
    endpoint_2 = track_info['endpoint_2']
    endpoint_1_frame = int(endpoint_1 * sample_rate)
    endpoint_2_frame = int(endpoint_2 * sample_rate)
    # track_path = "/homes/al4624/Documents/YuE_finetune/inference_testing_dataset/full_audio/078303.mp3"
    full_wave_signal = load_audio_mono(track_path).squeeze(0).unsqueeze(0)

    wave_signals = {
        "start": full_wave_signal[:, : endpoint_1_frame],
        "middle": full_wave_signal[:, endpoint_1_frame : endpoint_2_frame],
        "end": full_wave_signal[:, endpoint_2_frame : ],
    }

    probs = {}

    for segment, signal in wave_signals.items():
        prob = calc_tags_prob(model, signal)
        # print(f"[DEBUG] Segment {segment} probabilities: {prob}")
        probs[segment] = prob


    #start and end segments are the provided audio, middle is the generated transition
    kl_div_start = calc_KL_div(probs["start"], probs["middle"])
    kl_div_end = calc_KL_div(probs["end"], probs["middle"])

    # print(f"[DEBUG] KL divergence for track {track_name}: {kl_div}")

    return {
            "track_name": track_name,
            "kl_div_start": kl_div_start.item(),
            "kl_div_end": kl_div_end.item(),
            }


track_info_file_path = "/homes/al4624/Documents/YuE_finetune/inference_testing_dataset/track_info/info.json"

gen_track_dir_path = "/homes/al4624/Documents/YuE_finetune/inference_testing_dataset/full_audio"
gen_track_paths = [str(file) for file in Path(gen_track_dir_path).glob("*.mp3") if file.is_file()]

model = load_model(mode="logits").cuda()

track_kl_divs = []
for gen_track_path in gen_track_paths:
    track_kl_div = calc_KL_div_for_track(model, gen_track_path, track_info_file_path)
    track_kl_divs.append(track_kl_div)

json_obj = {"track_kl_divs": track_kl_divs}

with open("kl_divs.json", "w", encoding="utf-8") as f:
    json.dump(json_obj, f, indent=4)

print(track_kl_divs)



