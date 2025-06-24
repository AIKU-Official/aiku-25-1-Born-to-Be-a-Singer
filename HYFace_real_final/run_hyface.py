from os import listdir, path
import numpy as np
import argparse
import torch
import sys
# sys.path.append('HYFace_kgy_final')

from inference.run_app import main as hyface

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    # parser.add_argument('--main_model', type=str, default="/home/aikusrv03/lipsync/HYFace_kgy_final/pretrain/main.pth", help='your-HYFace-main-model-root')
    # parser.add_argument('--sub_model', type=str, default="/home/aikusrv03/lipsync/HYFace_kgy_final/pretrain/sub.pth", help='your-HYFace-sub-model-root')
    # parser.add_argument('--source', type=str, default='/home/aikusrv03/lipsync/data/Nell_source_16k.wav')
    # parser.add_argument('--target', type=str, default='/home/aikusrv03/lipsync/data/jiheon.jpg')
    # parser.add_argument('--output_path', type=str, default= '/home/aikusrv03/lipsync/output/converted.wav', help='output path for generated images')
    
    parser.add_argument('--main_model', type=str, default="pretrain/main.pth", help='your-HYFace-main-model-root')
    parser.add_argument('--sub_model', type=str, default="pretrain/sub.pth", help='your-HYFace-sub-model-root')
    parser.add_argument('--source', type=str, default='inference/Nell_source_16k.wav', help='your-source-audio-root')
    parser.add_argument('--target', type=str, default='/home/aikusrv03/lipsync/data/jiheon.jpg', help='your-target-img-root')
    parser.add_argument('--output_path', type=str, default='/home/aikusrv03/lipsync/output/converted.wav', help='your-target-output-root')
    parser.add_argument('--output_vocal_path', type=str, default='/home/aikusrv03/lipsync/output/converted_vocal.wav', help='your-target-vocal_output-root')

    opt = parser.parse_args()
    hyface(opt)