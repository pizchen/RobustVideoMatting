# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 22:12:33 2022

@author: olivec
"""

import sys
import os
import warnings
warnings.simplefilter("ignore", RuntimeWarning)

from inference import Converter

cd = os.path.dirname(os.path.realpath(__file__))
md = 'mobilenetv3'

if len(sys.argv) < 2:
    print("Please provide input images directory")
    sys.exit(1)

infile = sys.argv[1]

if not os.path.isfile(infile):
    print("File not exist: {}".format(infile))
    sys.exit(1)
    
imgcvt = Converter(md, os.path.join(cd, 'weight', 'rvm_'+md+'.pth'), 'cpu')

imgcvt.convert(
    input_source=infile,
    input_resize=None,
    downsample_ratio=None,
    output_type='video',
    output_composition='com_'+infile,
    output_alpha=None,
    output_foreground=None,
    output_video_mbps=1,
    seq_chunk=1,
    num_workers=0,
    progress=True
)