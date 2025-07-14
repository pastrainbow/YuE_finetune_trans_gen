import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))
from models.soundstream_hubert_new import SoundStream
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
from pathlib import Path
from omegaconf import OmegaConf
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


def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)
    
#bottom level/first layer encoding. This is sufficient since we don't need to train stage 2 model
def encode(audio_path, code_dir_path, codec_model, device):
    audio_data = load_audio_mono(audio_path)
    raw_codes = encode_audio(codec_model, audio_data, device, target_bw=0.5)
    code_file_name = os.path.splitext(os.path.basename(audio_path))[0] + ".npy"
    print(f"Finished encoding file {audio_path}")
    #dimension of the codes is (1, 1, n). We want to go out a level
    np.save(os.path.join(code_dir_path, code_file_name), raw_codes[0])
    #return raw_codes

# #no upsampling
# def decode(npy, save_path, codec_model, device):
#     tracks = []
#     codec_result = np.load(npy)
#     decodec_rlt=[]
#     with torch.no_grad():
#         decoded_waveform = codec_model.decode(torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(device))
#     decoded_waveform = decoded_waveform.cpu().squeeze(0)
#     decodec_rlt.append(torch.as_tensor(decoded_waveform))
#     decodec_rlt = torch.cat(decodec_rlt, dim=-1)
#     tracks.append(save_path)
#     save_audio(decodec_rlt, save_path, 16000)


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
# track_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large_sep"
# code_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large_codes"

track_dir_path = "/homes/al4624/Documents/YuE_finetune/test_sep_original"
code_dir_path = "/homes/al4624/Documents/YuE_finetune/test_codes"

track_paths = [str(file) for file in Path(track_dir_path).glob('*.mp3') if file.is_file()]

for track_path in track_paths:
    encode(track_path, code_dir_path, codec_model, device)


# #decode
# # reconstruct track
# npy = "/homes/al4624/Documents/YuE_finetune/YuE_finetune_trans_gen/finetune/example/npy/dummy.npy"
# save_path = "/homes/al4624/Documents/YuE_finetune/test_sep_original/test_reconstructed.mp3"
# #decode(npy, save_path, codec_model, device)


