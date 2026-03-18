import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact


# Load and Set up model
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

#tensorRT prep
model = model.half()
print("Model dtype and params halved:", next(model.parameters()).dtype)

dummy = torch.randn(1, 3, 640, 640).cuda().half()

torch.onnx.export(
   model,
   dummy,
   "realesrgan-v3-fp16.onnx",
   input_names=["input"],
   output_names=["output"],
   opset_version=18
)