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

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from models_original import (
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
        self.f = self.hyface_netc.to('cuda:0')
        self.hyface_netc.eval()
                
        self.hyface_netg = SynthesizerTrn(main_config.data.filter_length // 2 + 1,
                main_config.train.segment_size // main_config.data.hop_length,
                **main_config.model)
        self.hyface_netg, _, _, _ = utils.load_checkpoint(main_ckpt_path, self.hyface_netg, None)
        self.hyface_netg = self.hyface_netg.to('cuda:0')
        self.hyface_netg.eval()
        
        self.hyface_netf = F2F0(imgsize=112)
        self.hyface_netf, _, _, _ = utils.load_checkpoint(sub_ckpt_path, self.hyface_netf, None)
        self.hyface_netf = self.hyface_netf.to('cuda:0')
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
    parser.add_argument('--output_path', type=str, default="inference/result/original_hyface", help='your-target-output-root')
    base_args = parser.parse_args()

    # Setting
    main_config = load_config('configs/main.json')
    sub_config = load_config('configs/sub.json')
    hyface_conversion = HYFace_Conversion(main_config, sub_config, base_args.main_model, base_args.sub_model)

    auds = load_wav(base_args.source)
    imgs = load_img(base_args.target)
    inst = load_wav(base_args.inst).to("cuda:0")
    synth = hyface_conversion.synth_hyface(auds.to('cuda:0'), imgs.to('cuda:0'))
    
    min_len = min(synth.shape[-1], inst.shape[-1])
    final_mix = 1.2 * synth[:, :min_len].to(inst.device) + 0.8 * inst[:, :min_len]
    output_dir = f"inference/result/original_hyface"
    os.makedirs(output_dir, exist_ok=True) 
    singer_name = os.path.splitext(base_args.singer)[0]
    song_name = os.path.splitext(base_args.song_name)[0]
    torchaudio.save(f'{output_dir}/{singer_name}_{song_name}.wav', final_mix.cpu(), 16000)
    print(f"Successfully synthesized")
    
    #생성된보컬만저장
    pyannote_dir = os.path.join("inference", "result", "pyannote")
    os.makedirs(pyannote_dir, exist_ok=True)

    only_voice_path = f'{pyannote_dir}/{singer_name}_{song_name}_origin_onlyvoice.wav'
    torchaudio.save(only_voice_path, synth[:, :min_len].cpu(), 16000)
    print(f"[✔] original ver 생성된 순수 보컬 저장 완료 → {only_voice_path}")

# CUDA_VISIBLE_DEVICES=0 python inference/inference.py