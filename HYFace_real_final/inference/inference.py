import os
import sys
import json
import random
import argparse
import numpy as np

import librosa
import PIL
from PIL import Image
import torch
import torchaudio
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from transformers import HubertConfig, HubertModel

#추가 라이브러리
import torchaudio
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn, F2F0)
import utils
from utils import load_wav_to_torch

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

trans = transforms.Compose([transforms.Resize((112,112), interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(112), transforms.ToTensor()])  

#추가 코드
def extract_f0(audio, sampling_rate, hop_length=320, predictor="rmvpe", device="cuda:0"):
    f0_predictor = utils.get_f0_predictor(
        predictor,
        hop_length=hop_length,
        sampling_rate=sampling_rate,
        device=device,
        threshold=0.05,
    )
    # f0만 쓸 경우:
    f0, uv = f0_predictor.compute_f0_uv(audio.cpu().numpy())
    f0 = torch.from_numpy(f0).float().unsqueeze(0).to(device)
    uv = torch.from_numpy(uv).float().unsqueeze(0).to(device)
    return f0, uv

def load_config(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    args = utils.HParams(**config)
    return args

def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split)[0] for line in f]
  return filepaths_and_text

def load_img(img_dir):
    img = Image.open(img_dir)
    img_tensor = trans(img)
    return img_tensor.unsqueeze(0)

def load_wav(wav_dir):
    audio, sampling_rate = load_wav_to_torch(wav_dir)
    if audio.ndim ==2:
        audio = audio.mean(dim = 1)
    if sampling_rate != 16000:
        # 리샘플링
        audio = torchaudio.functional.resample(audio, orig_freq=sampling_rate, new_freq=16000)
    audio_norm = audio / 32768.0
    return audio_norm.unsqueeze(0)
    
class HYFace_Conversion(object):
    def __init__(self, main_config, sub_config, main_ckpt_path, sub_ckpt_path):
        self.build_hyface(main_config, sub_config, main_ckpt_path, sub_ckpt_path)

    def build_hyface(self, main_config, sub_config, main_ckpt_path, sub_ckpt_path):
        self.hyface_netc = HubertModelWithFinalProj.from_pretrained("lengyue233/content-vec-best")
        self.f = self.hyface_netc.to(f'cuda:{base_args.cuda}')
        self.hyface_netc.eval()
        
        self.hyface_netg = SynthesizerTrn(main_config.data.filter_length // 2 + 1,
                main_config.train.segment_size // main_config.data.hop_length,
                **main_config.model)
        self.hyface_netg, _, _, _ = utils.load_checkpoint(main_ckpt_path, self.hyface_netg, None)
        self.hyface_netg = self.hyface_netg.to(f'cuda:{base_args.cuda}')
        self.hyface_netg.eval()
        
        self.hyface_netf = F2F0(imgsize=112)
        self.hyface_netf, _, _, _ = utils.load_checkpoint(sub_ckpt_path, self.hyface_netf, None)
        self.hyface_netf = self.hyface_netf.to(f'cuda:{base_args.cuda}')
        self.hyface_netf.eval()

    def synth_hyface(self, source_c, target_f):
        source_c = self.hyface_netc(source_c)["last_hidden_state"]
        source_c = F.interpolate(source_c.transpose(-1,-2), source_c.shape[1]*2, mode="nearest")
        target_f0, _ = self.hyface_netf.infer(target_f)
        synth, _ = self.hyface_netg.infer(source_c, None, None, 0.35, avgf0=target_f0, face=target_f)
        return synth.detach().cpu().squeeze(0)
    
if __name__ == "__main__":
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) 
    warnings.simplefilter(action='ignore', category=UserWarning) 
    
    parser = argparse.ArgumentParser()
    # set parameters
    parser.add_argument('--main_model', type=str, default="pretrain/main.pth", help='your-HYFace-main-model-root')
    parser.add_argument('--sub_model', type=str, default="pretrain/sub.pth", help='your-HYFace-sub-model-root')
    parser.add_argument('--source', type=str, default='', help='your-source-audio-root')
    parser.add_argument('--target', type=str, default='', help='your-target-img-root')
    parser.add_argument('--inst', type=str, default='', help='your-inst-root')
    parser.add_argument('--song_name', type=str, default='', help='song_name')
    parser.add_argument('--singer', type=str, default='', help='singer')
    parser.add_argument('--cuda', type=str, default='0', help='GPU_ID')
    parser.add_argument('--output_path', type=str, default="inference/result/our_hyface", help='your-target-output-root')
    base_args = parser.parse_args()

    #보컬, 악기 분리
    auds = load_wav(base_args.source).to(f"cuda:{base_args.cuda}")
    inst = load_wav(base_args.inst).to(f"cuda:{base_args.cuda}")

    # Setting
    main_config = load_config('configs/main.json')
    sub_config = load_config('configs/sub.json')
    hyface_conversion = HYFace_Conversion(main_config, sub_config, base_args.main_model, base_args.sub_model)

    # --- content 추출
    with torch.no_grad():
        source_c = hyface_conversion.hyface_netc(auds)["last_hidden_state"]
        source_c = F.interpolate(source_c.transpose(-1,-2), source_c.shape[1]*2, mode="nearest")
    
    # --- f0 추출 (hop_length, predictor, device 등 config에서 조정)
    f0, uv = extract_f0(auds.squeeze(0), sampling_rate=16000, hop_length=320, predictor="rmvpe", device=f"cuda:{base_args.cuda}")
    # UV flag 등은 노래에서는 대부분 1로 두거나, 추후 zero crossing 등으로 보정 가능
    
    # --- target face emb (얼굴에서 임베딩 추출)
    imgs = load_img(base_args.target).to(f'cuda:{base_args.cuda}')
    target_face = imgs  # (1, 3, 112, 112)
    
    if source_c.shape[-1] != f0.shape[-1]:
        import torch.nn.functional as F
        f0 = F.interpolate(f0.unsqueeze(1), size=source_c.shape[-1], mode='nearest').squeeze(1)
        uv = F.interpolate(uv.unsqueeze(1), size=source_c.shape[-1], mode='nearest').squeeze(1)

    # --- generator로 변환
    with torch.no_grad():
        synth, _ = hyface_conversion.hyface_netg.infer(source_c, f0, None, avgf0=None, face=target_face)
            # synth: (N,) or (1, N) or (batch, channel, N) ... 등등 shape이 다를 수 있음
    if synth.ndim == 1:
        synth = synth.unsqueeze(0)  # (N,) -> (1, N)
    elif synth.ndim == 2:
        pass  # 이미 (1, N) or (C, N)
    elif synth.ndim == 3:
        synth = synth.squeeze(0)  # (1, 1, N) -> (1, N)
    
    #보컬, 악기 합성
    min_len = min(synth.shape[-1], inst.shape[-1])
    final_mix = 1.2 * synth[:, :min_len].to(inst.device) + 0.8 * inst[:, :min_len]
    output_dir = f"inference/result/our_hyface"
    os.makedirs(output_dir, exist_ok=True) 
    singer_name = os.path.splitext(base_args.singer)[0]
    song_name = os.path.splitext(base_args.song_name)[0]
    torchaudio.save(f'{output_dir}/{singer_name}_{song_name}.wav', final_mix.cpu(), 16000)
    print(f"Successfully synthesized")
    


    #생성된 보컬만 저장
    pyannote_dir = os.path.join("inference", "result", "pyannote")
    os.makedirs(pyannote_dir, exist_ok=True)

    only_voice_path = f'{pyannote_dir}/{singer_name}_{song_name}_onlyvoice.wav'
    torchaudio.save(only_voice_path, synth[:, :min_len].cpu(), 16000)
    print(f"[✔] 생성된 순수 보컬 저장 완료 → {only_voice_path}")

# CUDA_VISIBLE_DEVICES=0 python inference/inference.py