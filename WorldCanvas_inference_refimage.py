import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.WorldCanvas import WorldCanvasPipeline, ModelConfig
import os
import torch.distributed as dist
from safetensors.torch import load_file
import torch.nn as nn
import json
import imageio
import torchvision
import numpy as np
import cv2
import random
import re
import argparse

colors = np.array([(230, 25, 75), (67, 99, 216), (56, 195, 56), (255, 225, 25), (145, 30, 180), (70, 240, 240), (245, 130, 49)], dtype=np.uint8)
colors_names = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'orange']
special_names = ['+++', '@@@', '~~~', '$$$', '^^^', '&&&', '---']
if_color = 0
if_mask = 1
if_vr = 0
if_special_corr = 0
controls = ['gaussian_channel', 'vae_channel']
vae_channel = 'point'
T = 81
H = 480
W = 832
    
def create_circle_image(r, a, size=None):
    if size is None:
        size = int(2 * r) + 1

    image = np.zeros((size, size, 3), dtype=np.uint8)

    center = size // 2

    y, x = np.ogrid[:size, :size]
    
    dist_squared = (x - center)**2 + (y - center)**2

    mask = dist_squared <= r**2

    for i in range(3):
        image[:, :, i][mask] = a[i]

    return image

def crop_and_resize(image, target_height, target_width):
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    image = torchvision.transforms.functional.resize(
        image,
        (round(height*scale), round(width*scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )
    image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
    return image

def load_first_frame(file_path, height, width):
    reader = imageio.get_reader(file_path)
    frame = reader.get_data(0)
    frame = Image.fromarray(frame)
    frame = frame.convert('RGB')
    frame = crop_and_resize(frame, height, width)
    frames = [frame]
    reader.close()
    return frames[0]

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:  
        x = json.load(f)
    re_ = {}

    if '_crop.json' in file_path:
        if "id_caption_map" not in x[-2].keys():
            return None
        if len(x[-2]["id_caption_map"]) == 0:
            return None
    
    else:
        if "id_caption_map" not in x[-1].keys():
            return None
        if len(x[-1]["id_caption_map"]) == 0:
            return None

    if '_crop.json' in file_path:
        re_['crop'] = x[-1]
        x = x[:-1]

    id_caption_map = x[-1]["id_caption_map"]
    text_prompt = ''
    id_caption_order = {}
    color_idx = 0

    appears_ids = []

    re_['tracking_points'] = []
    re_['vis'] = []
    re_['rs'] = []
    re_['ids'] = []
    re_['point_masks'] = []

    for tp in x:
        if "tracking" in list(tp.keys()):
            re_['tracking_points'].append(tp['tracking'])
            if (tp['tracking'][0][0] < 0 or tp['tracking'][0][0] >= W) or (tp['tracking'][0][1] < 0 or tp['tracking'][0][1] >= H):
                appears_ids.append(str(tp['id']))
            re_['vis'].append(tp['tracking_vis_value'])
            re_['rs'].append(tp['r'])
            re_['ids'].append(tp['id'])
            re_['point_masks'].append(tp["mask_cluster"])

    if if_color == 1:
        id_color_map = {}
        color_n = list(range(7))
        random.shuffle(color_n)

        for tid, ca in id_caption_map.items():
            id_color_map[tid] = colors[color_n[color_idx]]
            text_prompt += colors_names[color_n[color_idx]]
            if tid in appears_ids:
                text_prompt += ' mask appears: '
            else:
                text_prompt += ' mask: '
            text_prompt += re.sub(r'\s+', ' ', ca).strip()
            text_prompt += os.linesep
            id_caption_order[tid] = color_idx
            color_idx += 1
            
        re_['id_color_map'] = id_color_map
    elif if_special_corr == 1:
        id_special_map = {}
        for tid, ca in id_caption_map.items():
            id_special_map[tid] = color_idx + 1
            if tid not in appears_ids:
                text_prompt += 'Object '
                text_prompt += special_names[color_idx]
                text_prompt += ' : '
            else:
                text_prompt += 'Object '
                text_prompt += special_names[color_idx]
                text_prompt += ' appears: '
            text_prompt += re.sub(r'\s+', ' ', ca).strip()
            text_prompt += os.linesep
            id_caption_order[tid] = color_idx
            color_idx += 1
        re_['id_special_map'] = id_special_map
    else:
        for tid, ca in id_caption_map.items():
            if tid not in appears_ids:
                text_prompt += f'Object {color_idx+1}: '
            else:
                text_prompt += f'Object {color_idx+1} appears: '
            text_prompt += re.sub(r'\s+', ' ', ca).strip()
            text_prompt += os.linesep
            id_caption_order[tid] = color_idx
            color_idx += 1

    re_['text_prompt'] = text_prompt[:-1]
    re_['id_caption_order'] = id_caption_order
    
    return re_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed")
    args = parser.parse_args()
    
    local_rank = 0
    device = f"cuda:{local_rank}"
    
    sample = ['examples/mouse_dog.jpg', 'examples/mouse_dog.json']  # [img_path, control_path]
    print(f"Loading model to {device}...")

    pipe = WorldCanvasPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
            model_configs=[
                ModelConfig(path="./checkpoints/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                ModelConfig(path="./checkpoints/WorldCanvas_dit/WorldCanvas_ref/high_model.safetensors", offload_device="cpu"),
                ModelConfig(path="./checkpoints/WorldCanvas_dit/WorldCanvas_ref/low_model.safetensors", offload_device="cpu"),
                ModelConfig(path="./checkpoints/Wan2.2-I2V-A14B/Wan2.1_VAE.pth", offload_device="cpu"),
                ],
    )
    pipe.enable_vram_management()
    print(f"Model loaded.")
 
    pipe.to(device)

    sample_json = load_json(sample[1])
    current_prompt = sample_json['text_prompt']
    
    input_image = load_first_frame(sample[0], H, W)

    video = pipe(
        prompt=current_prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        seed=args.seed,
        json_data=sample_json,
        tiled=True,
        height=H,
        width=W,
        num_frames=T,
        input_image=input_image,
        switch_DiT_boundary=0.9,
        num_inference_steps=50,
        if_color=if_color,
        if_mask=if_mask,
        if_special_corr=if_special_corr,
        if_vr=if_vr,
        controls=controls,
        vae_channel=vae_channel,
    )
    print(f"Video generation completed.")

    video_array = np.stack(video, axis=0)
    video_array_no_point = video_array.copy()

    for i, tp in enumerate(sample_json['tracking_points']):
        r = 10

        if if_color == 1:
            if str(sample_json['ids'][i]) in sample_json['id_color_map'].keys():
                color_traj_area = create_circle_image(r, sample_json['id_color_map'][str(sample_json['ids'][i])])
            else:
                color_traj_area = create_circle_image(r, np.array((255, 255, 255), dtype=np.uint8))
        else:
            color_traj_area = create_circle_image(r, np.array((255, 255, 255), dtype=np.uint8))

        for j in range(T):
            w, h = tp[j]
            if sample_json['vis'][i][j] > 0.5:
                x1 = int(max(w - r, 0))
                x2 = int(min(w + r, W - 1))
                y1 = int(max(h - r, 0))
                y2 = int(min(h + r, H - 1))
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue
                need_map = cv2.resize(color_traj_area, (x2-x1+1, y2-y1+1))
                video_array[j, y1:y2+1, x1:x2+1, :] = torch.tensor(need_map, dtype=torch.float32)
            
    pil_frames = [Image.fromarray(frame) for frame in video_array]
    pil_frames_no_point = [Image.fromarray(frame) for frame in video_array_no_point]
    
    output_folder = f"vis_jerry_{args.seed}"
    output_folder_no_point = f"novis_jerry_{args.seed}"

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder_no_point, exist_ok=True)

    output_path = os.path.join(output_folder, f"video_seed{args.seed}.mp4")
    output_path_no_point = os.path.join(output_folder_no_point, f"video_seed{args.seed}.mp4")
    save_video(pil_frames, output_path, fps=16, quality=5)
    save_video(pil_frames_no_point, output_path_no_point, fps=16, quality=5)
    print(f"Video saved to {output_path}")

    print(f"All tasks completed. Exiting.")


if __name__ == "__main__":
    main()