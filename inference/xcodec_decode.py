import os
import sys
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
def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

#no upsampling, no stage 2 model, so quality is not going to be amazing
def decode(npy, save_path, codec_model, device):
    tracks = []
    codec_result = np.load(npy)
    decodec_rlt=[]
    with torch.no_grad():
        decoded_waveform = codec_model.decode(torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(device))
    decoded_waveform = decoded_waveform.cpu().squeeze(0)
    decodec_rlt.append(torch.as_tensor(decoded_waveform))
    decodec_rlt = torch.cat(decodec_rlt, dim=-1)
    tracks.append(save_path)
    save_audio(decodec_rlt, save_path, 16000)


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


#decode
# reconstruct track
npy = "/homes/al4624/Documents/YuE_finetune/test_files/codes/130395.Instrumental.npy"
save_path = "/homes/al4624/Documents/YuE_finetune/test_files/code_reconstruction/130395.Instrumental.Reconstruction.mp3"
decode(npy, save_path, codec_model, device)



