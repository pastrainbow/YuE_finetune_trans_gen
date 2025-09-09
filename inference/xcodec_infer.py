import os
import sys
import argparse
os.environ['TRANSFORMERS_CACHE'] = '/vol/bitbucket/al4624/cache/general_cache/transformer_cache'
os.environ['HF_HOME'] = '/vol/bitbucket/al4624/cache/general_cache/hf_home_cache'
os.environ['XDG_CACHE_HOME'] = '/vol/bitbucket/al4624/cache/general_cache/xdg_cache_home'
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))
from models.soundstream_hubert_new import SoundStream
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
from pathlib import Path
from omegaconf import OmegaConf
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from mmtokenizer import _MMSentencePieceTokenizer
def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio

def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
    if len(audio_prompt.shape) < 3:
        audio_prompt.unsqueeze_(0)
    with torch.no_grad():
        raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
    raw_codes = raw_codes.transpose(0, 1)
    raw_codes = raw_codes.cpu().numpy().astype(np.int16)
    return raw_codes[0]

#bottom level/first layer encoding. This is sufficient since we don't need to train stage 2 model
def encode(audio_path, code_dir_path, codec_model, device, target_bw=0.5):
    try:
        audio_data = load_audio_mono(audio_path)
        raw_codes = encode_audio(codec_model, audio_data, device, target_bw)
        code_file_name = os.path.splitext(os.path.basename(audio_path))[0] + ".npy"
        print(f"Finished encoding file {audio_path}")
        np.save(os.path.join(code_dir_path, code_file_name), raw_codes)
    except Exception as e:
        print(f"Error encoding {audio_path}: {e}. Skipping")
        raise


def noise_gen_gaussian(range_factor, frame_count, device):
    mean = 0.0
    #portion of values in range = 1 - 1 / range_factor^2
    #value range is 1 here
    std = 1.0 / range_factor
    
    # Gaussian noise: create a random normal distribution that has the same size as the data to add noise to 
    # Genearte noise with same size as that of the data.
    return torch.normal(mean=mean, std=std, size=(frame_count,), device=device)

#signal weight controls how much of the audio signal we want to keep
def add_noise(audio_data, noise_data, signal_weight):
    audio_data *= signal_weight
    audio_data += noise_data * (1.0 - signal_weight)

def noise_file(file_path, signal_weight, device, sample_rate = 16000):
    try:
        audio_data = load_audio_mono(file_path).to(device)[0]
        frame_count = len(audio_data)
        #split to 3 segments: start, middle and end
        segment_frame_count = frame_count // 3
        middle_segment_start = segment_frame_count
        middle_segment_end = segment_frame_count * 2
        audio_data_middle = audio_data[middle_segment_start : middle_segment_end]
        
        #Add the noise to the data
        #To save computation cost, we can also generate the noise only once, and then use slices of the same noise throughout training to accomodate
        #for different durations
        #range factor of 4 covers the dynmaic range quite well without clipping too much
        noise_data = noise_gen_gaussian(4, segment_frame_count, device=device)
        #0.9 signal weight for now, first finetune introduces only a small amount of noise 
        add_noise(audio_data_middle, noise_data, signal_weight)
        
        #clip above and below, avoid out of range values
        audio_data_middle.clamp_(-1.0, 1.0)

        print(f"File {file_path} finished noising. Middle segement starts at {middle_segment_start / sample_rate}, ends at {middle_segment_end / sample_rate} ")
        return audio_data.unsqueeze(0).to(dtype=torch.float32)
    #FMA dataset has corrupted files. It is normal for a few files to fail the processing.
    except Exception as e:
        print(f"Error processing {file_path}: {e}. Skipping")
        raise


def noise_encode(audio_path, signal_weight, code_dir_path, codec_model, device, target_bw=0.5):
    try:
        if (not os.path.exists(audio_path)):
            print(f"File {audio_path} does not exist. Skipping")
            return
        audio_data = noise_file(audio_path, signal_weight, device)
        raw_codes = encode_audio(codec_model, audio_data, device, target_bw)
        code_file_name = os.path.splitext(os.path.basename(audio_path))[0] + ".noised.npy"
        print(f"Finished noising and encoding file {audio_path}")
        np.save(os.path.join(code_dir_path, code_file_name), raw_codes)
    except Exception as e:
        print(f"Error noising encoding {audio_path}: {e}. Skipping")
        raise


#determine encoding bandwidth
parser = argparse.ArgumentParser()
parser.add_argument("--num_quantizers", type=int, default=1, help="Number of quantizer layers to use for encoding")
parser.add_argument("--inst", action='store_true', help="Whether to encode instrumental tracks")
parser.add_argument("--vocals", action='store_true', help="Whether to encode vocal tracks")
parser.add_argument("--mixture", action='store_true', help="Whether to encode mixture tracks")
parser.add_argument("--noised_inst", action='store_true', help="Whether to encode noise instrumental tracks")
parser.add_argument("--noise_levels", nargs="+", type=str, help="The list of noise levels to use for encoding") #we don't use float here to avoid precision issue

args = parser.parse_args()
encoder_bandwidth = args.num_quantizers * 0.5 #assuming 16000 Hz sample rate and 320 hop length, which is the original config for Xcodec
print(f"[DEBUG] encoder target bandwidth is {encoder_bandwidth} kbps, with {args.num_quantizers} quantizer layers.")

#initialise model
cuda_idx = 0
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
basic_model_config = "./xcodec_mini_infer/final_ckpt/config.yaml"
resume_path = "./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth"
model_config = OmegaConf.load(basic_model_config)
codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
parameter_dict = torch.load(resume_path, map_location='cpu', weights_only=False)
codec_model.load_state_dict(parameter_dict['codec_model'])
codec_model.to(device)
codec_model.eval()

#encode
sep_track_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large_sep"
sep_code_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large_sep_codes"
sep_track_paths = [str(file) for file in Path(sep_track_dir_path).glob('*.mp3') if file.is_file()]
num_sep_track = len(sep_track_paths)

mixture_code_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large_mixture_codes"

mixture_track_dir_paths = []

signal_weights = []

for noise_level in args.noise_levels:
    if noise_level not in ['0.1', '0.3', '0.5', '0.7', '1.0']:
        raise ValueError(f"Unsupported noise level {noise_level}. Supported levels are 0.1, 0.3, 0.5, 0.7, 1.0")
    mixture_track_dir_paths.append(f"/vol/bitbucket/al4624/finetune_dataset/fma_large/sep/noise_{noise_level}")
    signal_weights.append(1.0 - float(noise_level))

print(f"[DEBUG] mixture dir paths: {mixture_track_dir_paths}")
print(f"[DEBUG] signal weights: {signal_weights}")

inst_track_dir_path = sep_track_dir_path #modify to specify the directory for the instrumental audio tracks
noised_inst_code_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large_inst_noised_codes" #modify to specify the directory for outputting the noised instrumental codes

if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:

        #separated track encode
        # sep_encode_futures = [
        #     executor.map(encode, sep_track_paths, [sep_code_dir_path] * num_sep_track, [codec_model] * num_sep_track, [device] * num_sep_track, [encoder_bandwidth] * num_sep_track)
        # ]

      
        #instrumental track noising + encode
        for i in range(len(signal_weights)):
            signal_weight = signal_weights[i]
            mixture_track_dir_path = mixture_track_dir_paths[i]

            mixture_track_files = Path(mixture_track_dir_path).glob('*.mp3')

            inst_track_paths = [os.path.join(inst_track_dir_path, file.stem + ".Instrumental.mp3") for file in mixture_track_files if file.is_file()]
            num_inst_track = len(inst_track_paths)

            vocal_track_paths = [os.path.join(sep_track_dir_path, file.stem + ".Vocals.mp3") for file in mixture_track_files if file.is_file()]
            num_vocal_track = len(vocal_track_paths)

            mixture_track_paths =[str(file) for file in mixture_track_files if file.is_file()]

            if args.inst:
                executor.map(encode, inst_track_paths, [sep_code_dir_path] * num_inst_track, [codec_model] * num_inst_track, [device] * num_inst_track, [encoder_bandwidth] * num_inst_track)
            
            if args.vocals:
                executor.map(encode, vocal_track_paths, [sep_code_dir_path] * num_vocal_track, [codec_model] * num_vocal_track, [device] * num_vocal_track, [encoder_bandwidth] * num_vocal_track)

            if args.noised_inst:
                executor.map(noise_encode, inst_track_paths, [signal_weight] * num_inst_track, [noised_inst_code_dir_path] * num_inst_track, [codec_model] * num_inst_track, [device] * num_inst_track, [encoder_bandwidth] * num_inst_track)

            if args.mixture:
                executor.map(encode, mixture_track_paths, [mixture_code_dir_path] * num_sep_track, [codec_model] * num_sep_track, [device] * num_sep_track, [encoder_bandwidth] * num_sep_track)
                




