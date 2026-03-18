import torch
import cv2
import numpy as np
import os.path as osp
import glob
import shutil
import os
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from tqdm import tqdm

# TensorRT setup
TRT_ENGINE_PATH = 'realesrgan-v3-fp16.trt'
INPUT_H, INPUT_W = 640, 640
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_trt_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def trt_infer(context, input_img, stream, d_input, d_output, output_shape):
    # input_img: float16 numpy array, shape (1, 3, H, W)
    np.copyto(d_input.host, input_img.ravel())
    cuda.memcpy_htod_async(d_input.device, d_input.host, stream)
    context.set_tensor_address('input',  int(d_input.device))
    context.set_tensor_address('output', int(d_output.device))

    context.execute_async_v3(stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(d_output.host, d_output.device, stream)
    stream.synchronize()
    return d_output.host.reshape(output_shape)

class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

def allocate_buffers(engine):
    input_shape  = (1, 3, INPUT_H, INPUT_W)
    output_shape = (1, 3, INPUT_H * 4, INPUT_W * 4)  # x4 upscale

    d_input = HostDeviceMem(
        cuda.pagelocked_empty(int(np.prod(input_shape)),  dtype=np.float16),
        cuda.mem_alloc(int(np.prod(input_shape))  * np.dtype(np.float16).itemsize)
    )
    d_output = HostDeviceMem(
        cuda.pagelocked_empty(int(np.prod(output_shape)), dtype=np.float16),
        cuda.mem_alloc(int(np.prod(output_shape)) * np.dtype(np.float16).itemsize)
    )
    return d_input, d_output, output_shape

# load engine
engine  = load_trt_engine(TRT_ENGINE_PATH)
context = engine.create_execution_context()
stream  = cuda.Stream()
d_input, d_output, output_shape = allocate_buffers(engine)

print(f"TensorRT engine loaded: {TRT_ENGINE_PATH}")

# preprocess / postprocess
def preprocess(img):
    # img: BGR uint8 HxWx3 → float16 1x3xHxW in [0,1]
    img = cv2.resize(img, (INPUT_W, INPUT_H))
    img = img[:, :, ::-1].astype(np.float16) / 255.0   # BGR→RGB, normalise
    img = np.transpose(img, (2, 0, 1))[np.newaxis]      # HWC → 1CHW
    return np.ascontiguousarray(img)

def postprocess(output):
    # output: float16 1x3xH'xW' - BGR
    output = np.squeeze(output, axis=0)            # 3xH'xW'
    output = np.clip(output, 0, 1)
    output = np.transpose(output, (1, 2, 0))             # HWC
    output = (output * 255.0).astype(np.uint8)
    output = output[:, :, ::-1]                          # RGB - BGR
    return output

# pipeline functions
def extract_frames(video_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    idx = 0
    for path in glob.glob(video_path):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]

        if os.path.exists(path):
            cap = cv2.VideoCapture(path)

            if cap is not None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                frame_count = 0
                with tqdm(total=total_frames, desc="Extracting frames") as pbar:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        save_location = osp.join(output_path, f"frame_{frame_count:06d}.jpg")
                        cv2.imwrite(save_location, frame)
                        frame_count += 1
                        pbar.update(1)

            cap.release()
            return total_frames, fps

def upscale_frames(input_path, output_path):
    inference_times = []
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    with tqdm(total=len(glob.glob(input_path)), desc="Upscaling frames") as pbar:
        for path in glob.glob(input_path):
            base = osp.splitext(osp.basename(path))[0]

            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_COLOR)

                if img is not None:
                    inp = preprocess(img)

                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    out = trt_infer(context, inp, stream, d_input, d_output, output_shape)
                    torch.cuda.synchronize()
                    end = time.perf_counter()

                    inference_times.append(end - start)
                    output_img = postprocess(out)

                    save_location = osp.join(output_path, f"{base}.jpg")
                    cv2.imwrite(save_location, output_img)
                else:
                    print('Failed to upscale frame')
            else:
                print('Specified path is empty')

            pbar.update(1)

    avg_time = np.mean(inference_times)
    fps = 1.0 / avg_time
    print(f"Average inference time: {avg_time*1000:.2f}ms")
    print(f"Theoretical FPS: {fps:.2f}")

def create_video_from_frames(input_dir, temp_dir, output_path, fps):
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    frames = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))])
    print(f"Found {len(frames)} frames in {input_dir}")

    if not frames:
        raise ValueError("No frames found in directory")

    first_frame = cv2.imread(os.path.join(input_dir, frames[0]))
    if first_frame is None:
        raise ValueError(f"Failed to read first frame")

    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video  = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not video.isOpened():
        raise ValueError(f"Failed to open video writer for {output_path}")

    for i, frame_file in enumerate(frames):
        frame = cv2.imread(os.path.join(input_dir, frame_file))
        if frame is None:
            print(f"Warning: Failed to read frame: {frame_file}")
            continue
        video.write(frame)
        if i % 100 == 0:
            print(f"Processed {i}/{len(frames)} frames")

    video.release()
    shutil.rmtree(input_dir)
    shutil.rmtree(temp_dir)
    print(f"Video successfully created at {output_path}")

#  paths
low_res_video_path     = './LR/*'
video_frames_output_path = './temp'
load_frames_path       = './temp/*'
path_to_upscaled_frames  = './upscaled_frames'
output_video           = './final_output/new_video.mp4'
frames_per_second      = 24.0

#  pipeline
extract_frames(low_res_video_path, video_frames_output_path)
upscale_frames(load_frames_path, path_to_upscaled_frames)
create_video_from_frames(path_to_upscaled_frames,video_frames_output_path , output_video, frames_per_second)