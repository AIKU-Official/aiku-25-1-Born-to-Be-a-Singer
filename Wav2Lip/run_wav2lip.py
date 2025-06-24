import os
import numpy as np
import argparse
import torch
import sys
# sys.path.append('Wav2Lip')

from inference import main as wav2lip


if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='device for test')
    parser.add_argument('--checkpoint_path', type=str, default ='checkpoints/Wav2Lip/wav2lip_gan.pth',
						help='Name of saved checkpoint to load weights from')
    parser.add_argument('--face', type=str,  default='/home/aikusrv03/lipsync/data/jiheon.jpg',
						help='Filepath of video/image that contains faces to use')
    parser.add_argument('--audio', type=str,  default='/home/aikusrv03/lipsync/output/converted.wav',
						help='Filepath of video/audio file to use as raw audio source')
    parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
									 default='/home/aikusrv03/lipsync/output/result.avi')
    parser.add_argument('--static', type=bool, 
						help='If True, then use only first video frame for inference', default=False)
    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
						default=25., required=False)
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
						help='Padding (top, bottom, left, right). Please adjust to include chin at least')
    parser.add_argument('--face_det_batch_size', type=int, 
						help='Batch size for face detection', default=16)
    parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)
    parser.add_argument('--resize_factor', default=1, type=int, 
				help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
						help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
						'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')
    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
						help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
						'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
    parser.add_argument('--rotate', default=False, action='store_true',
						help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
						'Use if you get a flipped result, despite feeding a normal looking video')
    parser.add_argument('--nosmooth', default=False, action='store_true',
						help='Prevent smoothing face detections over a short temporal window')
    args = parser.parse_args()
    args.img_size = 96
    if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        args.static = True
    
    wav2lip(args)