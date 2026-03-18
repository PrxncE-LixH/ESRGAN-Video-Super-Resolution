from sympy import fps
import torch
import cv2
import numpy as np
import os.path as osp
import glob
import torch
import os
import time
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils import RealESRGANer
from tqdm import tqdm

# Set up model
model_path = 'realesr-general-x4v3.pth'

state_dict = torch.load(model_path, map_location=torch.device('cuda'))['params']
model = SRVGGNetCompact(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_conv=32,
    upscale=4,
    act_type='prelu'
)

model.load_state_dict(state_dict, strict=True)
model = model.cuda()
model.eval()

print("Original model dtype and params:", next(model.parameters()).dtype)

#tensoRT
dummy = torch.randn(1, 3, 256, 256).cuda()

torch.onnx.export(
    model,
    dummy,
    "realesrgan-v3.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {2: "h", 3: "w"},
        "output": {2: "H", 3: "W"}
    },
    opset_version=18
)
