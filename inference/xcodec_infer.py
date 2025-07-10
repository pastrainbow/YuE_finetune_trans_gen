import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))
from models.soundstream_hubert_new import SoundStream
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
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
    #dimension of the codes is (1, 1, n). We want to go out a level
    np.save(os.path.join(code_dir_path, code_file_name), raw_codes[0])
    #return raw_codes

#no upsampling
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

#encode
audio_path = "/homes/al4624/Documents/YuE_finetune/test_sep_original/test.mp3"
code_dir_path = "/homes/al4624/Documents/YuE_finetune/test_sep_original/"
encode(audio_path, code_dir_path, codec_model, device)

#decode
# reconstruct track
npy = "/homes/al4624/Documents/YuE_finetune/YuE_finetune_trans_gen/finetune/example/npy/dummy.npy"
save_path = "/homes/al4624/Documents/YuE_finetune/test_sep_original/test_reconstructed.mp3"
#decode(npy, save_path, codec_model, device)


