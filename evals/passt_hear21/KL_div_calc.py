import os
os.environ['HF_HOME'] = '/vol/bitbucket/al4624/cache/general_cache/hf_home_cache'
os.environ['XDG_CACHE_HOME'] = '/vol/bitbucket/al4624/cache/general_cache/xdg_cache_home'
import torch
import torchaudio
from torchaudio.transforms import Resample
import torch.nn.functional as F
from hear21passt.base30sec import load_model
import json

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
    print(f"CUDA signal: {wave_signal}")
    logits = model(wave_signal.cuda())
    #pytorch's KL divergence function accepts log softmax activated probabilities
    probs = F.log_softmax(logits, dim=1)
    print(f"sum: {torch.sum(probs)}")
    return probs


def calc_KL_div(prob_1, prob_2):
    return F.kl_div(prob_1, prob_2, reduction='batchmean')


track_name = "078303"
track_info_file_path = "/homes/al4624/Documents/YuE_finetune/inference_testing_dataset/track_info/info.json"

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
endpoint_1 = track_info['endpoint_1']
endpoint_2 = track_info['endpoint_2']
endpoint_1_frame = int(endpoint_1 * sample_rate)
endpoint_2_frame = int(endpoint_2 * sample_rate)
track_path = "/homes/al4624/Documents/YuE_finetune/inference_testing_dataset/full_audio/078303.mp3"
full_wave_signal = load_audio_mono(track_path).squeeze(0).unsqueeze(0)

wave_signals = {
    "start": full_wave_signal[:, : endpoint_1_frame],
    "middle": full_wave_signal[:, endpoint_1_frame : endpoint_2_frame],
    "end": full_wave_signal[:, endpoint_2_frame : ],
}

# model = load_model(mode="logits").cuda()

probs = {}

for segment, signal in wave_signals.items():
    print(f"Processing segment {segment} with signal {signal}")
    print(f"CUDA signal: {signal.cuda()}")
    # prob = calc_tags_prob(model, signal)
    # probs[segment] = prob

# kl_div_start = calc_KL_div(probs["start"], probs["middle"])
# kl_div_end = calc_KL_div(probs["middle"], probs["end"])

# kl_div = (kl_div_start + kl_div_end) / 2

# print(f"KL divergence: {kl_div}")

