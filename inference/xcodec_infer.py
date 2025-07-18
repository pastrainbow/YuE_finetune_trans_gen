import os
import sys
import argparse
os.environ['TRANSFORMERS_CACHE'] = '/vol/bitbucket/al4624/transformer_cache'
os.environ['HF_HOME'] = '/vol/bitbucket/al4624/hf_home_cache'
os.environ['XDG_CACHE_HOME'] = '/vol/bitbucket/al4624/xdg_cache_home'
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


def noise_gen_gaussian(range_factor, frame_count):
    mean = 0.0
    #portion of values in range = 1 - 1 / range_factor^2
    #value range is 1 here
    std = 1.0 / range_factor
    
    # Gaussian noise: create a random normal distribution that has the same size as the data to add noise to 
    # Genearte noise with same size as that of the data.
    return np.random.normal(mean, std, frame_count)

#signal weight controls how much of the audio signal we want to keep
def add_noise(audio_data, noise_data, signal_weight):
    audio_data *= signal_weight
    audio_data += noise_data * (1.0 - signal_weight)

def noise_file(file_path, signal_weight, sample_rate = 16000):
    try:
        #mono conversion - YuE only supports mono audio
        audio_data = load_audio_mono(file_path)[0].numpy()
        frame_count = len(audio_data)
        #if training examples are of the same length, we can just calculate this once for the first training example
        #split to 3 segments: start, middle and end
        segment_frame_count = int(frame_count / 3)
        middle_segment_start = segment_frame_count
        middle_segment_end = segment_frame_count * 2
        audio_data_middle = audio_data[middle_segment_start : middle_segment_end]
        
        #Add the noise to the data
        #To save computation cost, we can also generate the noise only once, and then use slices of the same noise throughout training to accomodate
        #for different durations
        #range factor of 4 covers the dynmaic range quite well without clipping too much
        noise_data = noise_gen_gaussian(4, segment_frame_count)
        #0.9 signal weight for now, first finetune introduces only a small amount of noise 
        add_noise(audio_data_middle, noise_data, signal_weight)
        
        #clip above and below, avoid out of range values
        np.clip(audio_data_middle, -1.0, 1.0, out = audio_data_middle)

        print(f"File {file_path} finished noising. Middle segement starts at {middle_segment_start / sample_rate}, ends at {middle_segment_end / sample_rate} ")
        return torch.from_numpy(np.array([audio_data]))
    #FMA dataset has corrupted files. It is normal for a few files to fail the processing.
    except Exception as e:
        print(f"Error processing {file_path}: {e}. Skipping")
        raise

def noise_encode(audio_path, signal_weight, code_dir_path, codec_model, device, target_bw=0.5):
    try:
        if (not os.path.exists(audio_path)):
            print(f"File {audio_path} does not exist. Skipping")
            return
        audio_data = noise_file(audio_path, signal_weight)
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
args = parser.parse_args()
encoder_bandwidth = args.num_quantizers * 0.5 #assuming 16000 Hz sample rate and 320 hop length
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
# sep_track_dir_path = "/homes/al4624/Documents/YuE_finetune/test_sep_original"
# sep_code_dir_path = "/homes/al4624/Documents/YuE_finetune/test_codes"
sep_track_paths = [str(file) for file in Path(sep_track_dir_path).glob('*.mp3') if file.is_file()]
num_sep_track = len(sep_track_paths)

# mixture_track_dir_paths = ["/vol/bitbucket/al4624/finetune_dataset/fma_large/sep/noise_0.3", 
#                             "/vol/bitbucket/al4624/finetune_dataset/fma_large/sep/noise_0.5",
#                             "/vol/bitbucket/al4624/finetune_dataset/fma_large/sep/noise_0.7",
#                             "/vol/bitbucket/al4624/finetune_dataset/fma_large/sep/noise_1.0"]

mixture_track_dir_paths = ["/vol/bitbucket/al4624/finetune_dataset/fma_large/sep/noise_1.0"] #modify to specify the tracks to encode

# signal_weights = [0.7, 0.5, 0.3, 0.0]

signal_weights = [0.0] #modify to specify the noise levels. Must contain equal number of elements as mixture_track_dir_paths

inst_track_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large_sep"
inst_code_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large_inst_noised_codes"
# inst_track_paths = [str(file) for file in Path(inst_track_dir_path).glob('*.Instrumental.mp3') if file.is_file()]
# num_inst_track = len(inst_track_paths)


if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        # encode_futures = [
        #     executor.map(encode, sep_track_paths, [sep_code_dir_path] * num_sep_track, [codec_model] * num_sep_track, [device] * num_sep_track, [encoder_bandwidth] * num_inst_track)
        # ]

        for i in range(len(signal_weights)):
            signal_weight = signal_weights[i]
            mixture_track_dir_path = mixture_track_dir_paths[i]
            inst_track_paths = [os.path.join(inst_track_dir_path, file.stem + ".Instrumental.mp3") for file in Path(mixture_track_dir_path).glob('*.mp3') if file.is_file()]
            num_inst_track = len(inst_track_paths)
            executor.map(noise_encode, inst_track_paths, [signal_weight] * num_inst_track, [inst_code_dir_path] * num_inst_track, [codec_model] * num_inst_track, [device] * num_inst_track, [encoder_bandwidth] * num_inst_track)

        # noise_encode_futures = [
        #     executor.map(noise_encode, inst_track_paths, [0.9] * num_inst_track, [inst_code_dir_path] * num_inst_track, [codec_model] * num_inst_track, [device] * num_inst_track, [encoder_bandwidth] * num_inst_track)
        # ]



