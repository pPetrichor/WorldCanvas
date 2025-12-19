import imageio, os, torch, warnings, torchvision, argparse, json, re, math
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import numpy as np
import cv2
import random

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video", "json_path"),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        fps=16,
        if_color=0,
        if_mask=1,
        if_special_corr=0,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            if_color = args.if_color
            if_mask = args.if_mask
            if_special_corr = args.if_special_corr
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        self.fps = fps
        self.if_color = if_color
        self.if_mask = if_mask
        self.if_special_corr = if_special_corr
        self.colors = np.array([(230, 25, 75), (67, 99, 216), (56, 195, 56), (255, 225, 25), (145, 30, 180), (70, 240, 240), (245, 130, 49)], dtype=np.uint8)
        self.colors_names = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'orange']
        self.special_names = ['+++', '@@@', '~~~', '$$$', '^^^', '&&&', '---']
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        orig_fps = float(reader.get_meta_data()['fps'])
        num_frames = int(reader.count_frames())
        factor = orig_fps / self.fps
        frame_in_idx = 0 
        frame_out_idx = 0 
        accumulator = 0.0 
        ret_frames = []

        if factor >= 1:
            while frame_out_idx < self.num_frames and frame_in_idx < num_frames:
                accumulator += 1
                if accumulator >= factor:
                    accumulator -= factor           
                    frame_out_idx += 1
                    ret_frames.append(frame_in_idx)

                frame_in_idx += 1
        else:
            duration = num_frames / orig_fps
            target_total_frames = int(np.round(self.fps * duration))
            target_times = np.linspace(0, duration, target_total_frames, endpoint=False)
            frame_indices = np.round(target_times * orig_fps).astype(int)
            frame_indices = np.clip(frame_indices, 0, num_frames - 1)
            ret_frames = frame_indices.tolist()
            if len(ret_frames) > self.num_frames:
                ret_frames = ret_frames[:self.num_frames]

        ret_len = len(ret_frames)
        if ret_len < self.num_frames:
            while ret_len > 1 and ret_len % self.time_division_factor != self.time_division_remainder:
                ret_len -= 1
        return ret_frames[:ret_len]
    

    def load_video(self, file_path):    
        reader = imageio.get_reader(file_path)
        frames_ids = self.get_num_frames(reader)
        frames = []
        for frame_id in frames_ids:
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
        
    def load_json(self, file_path):
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
        id_caption_map = dict(random.sample(list(id_caption_map.items()), len(id_caption_map)))
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
                if (tp['tracking'][0][0] < 0 or tp['tracking'][0][0] >= self.width) or (tp['tracking'][0][1] < 0 or tp['tracking'][0][1] >= self.height):
                    appears_ids.append(str(tp['id']))
                re_['vis'].append(tp['tracking_vis_value'])
                re_['rs'].append(tp['r'])
                re_['ids'].append(tp['id'])
                re_['point_masks'].append(tp["mask_cluster"])

        if self.if_color == 1:
            id_color_map = {}
            color_n = list(range(7))
            random.shuffle(color_n)

            for tid, ca in id_caption_map.items():
                id_color_map[tid] = self.colors[color_n[color_idx]]
                text_prompt += self.colors_names[color_n[color_idx]]
                if tid in appears_ids:
                    text_prompt += ' mask appears: '
                else:
                    text_prompt += ' mask: '
                text_prompt += re.sub(r'\s+', ' ', ca).strip()
                text_prompt += os.linesep
                id_caption_order[tid] = color_idx
                color_idx += 1
                
            re_['id_color_map'] = id_color_map
        elif self.if_special_corr == 1:
            id_special_map = {}
            for tid, ca in id_caption_map.items():
                id_special_map[tid] = color_idx + 1
                if tid not in appears_ids:
                    text_prompt += 'Object '
                    text_prompt += self.special_names[color_idx]
                    text_prompt += ' : '
                else:
                    text_prompt += 'Object '
                    text_prompt += self.special_names[color_idx]
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
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        elif file_path.split(".")[-1] == 'json':
            return self.load_json(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        try:
            for key in self.data_file_keys:
                if key in data:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                    if data[key] is None:
                        warnings.warn(f"cannot load file {data[key]}.")
                        return None
            
            if 'crop' in data['json_path'].keys():
                frames_new = []
                crop_related = data['json_path']['crop']
                y0, x0 = crop_related["anchor_yx"]
                crop_h, crop_w = crop_related["crop_size_hw"]
                for frame in data['video']:
                    frame = np.array(frame, dtype=np.float32)
                    x1 = min(x0 + crop_w, self.width)
                    y1 = min(y0 + crop_h, self.height)
                    crop = frame[y0:y1, x0:x1]

                    if crop.shape[1] != crop_w or crop.shape[0] != crop_h:
                        assert 0
                    resized = cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                    resized = np.clip(resized, 0, 255).astype(np.uint8)
                    frames_new.append(Image.fromarray(resized))
                data['video'] = frames_new

        except Exception as e:
            print(e)
            return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict



class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, validation_config=None, save_every_n_steps=1000):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.validation_config = validation_config
        self.save_every_n_steps = save_every_n_steps
        # Create subdirectories for clarity
        self.resumable_path = os.path.join(output_path, "resumable")
        self.portable_path = os.path.join(output_path, "portable")
        self.validation_path = os.path.join(output_path, "validation")
        os.makedirs(self.resumable_path, exist_ok=True)
        os.makedirs(self.portable_path, exist_ok=True)
        os.makedirs(self.validation_path, exist_ok=True)


    def on_step_end(self, accelerator, model, global_step):
        self.save_model(accelerator, model, f"step-{global_step}")


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}")
            accelerator.save(state_dict, path, safe_serialization=True)


    # def on_training_end(self, accelerator, model, save_steps=None):
    #     if save_steps is not None and self.num_steps % save_steps != 0:
    #         self.save_model(accelerator, model, f"step-{self.num_steps}")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()

        path = os.path.join(self.output_path, file_name)

        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
        accelerator.wait_for_everyone()
        accelerator.save_state(path)


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    resume_from_checkpoint: str = None,
    mixed_precision: str = "bf16",
    enabled_deepspeed: bool = False,
    model_ds_config: str = None,
):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    if enabled_deepspeed:
        from accelerate.utils import DeepSpeedPlugin
        accelerator = Accelerator(
                deepspeed_plugins=DeepSpeedPlugin(hf_ds_config=model_ds_config),
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
        if accelerator.is_main_process:
            print("Setting up deepspeed zero2 optimization done.")
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
        )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    global_step = 0
    if resume_from_checkpoint is not None:
        accelerator.load_state(resume_from_checkpoint)
        global_step = int(re.search(r"step-(\d+)", resume_from_checkpoint).group(1))
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    starting_epoch = global_step // num_update_steps_per_epoch if num_update_steps_per_epoch > 0 else 0
    
    for epoch_id in range(starting_epoch, num_epochs):
        for data in tqdm(dataloader, desc=f"Epoch {epoch_id}"):
            if data is None:
                continue
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                # model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
                if accelerator.sync_gradients:
                    global_step += 1
                    if global_step == 0 or global_step % save_steps == 0:
                        model_logger.on_step_end(accelerator, model, global_step)
                    
                if global_step == 60000:
                    break
        # if save_steps is None:
        #     model_logger.on_epoch_end(accelerator, model, epoch_id)
        if global_step == 60000:
            break
    # model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(model: DiffusionTrainingModule, dataset, output_path="./models"):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0])
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    os.makedirs(os.path.join(output_path, "data_cache"), exist_ok=True)
    for data_id, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = model.forward_preprocess(data)
            inputs = {key: inputs[key] for key in model.model_input_keys if key in inputs}
            torch.save(inputs, os.path.join(output_path, "data_cache", f"{data_id}.pth"))



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="video, json_path", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=45, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--controls", type=str, default="control_latents, gaussian_channel, vae_channel", help="controls.")
    parser.add_argument("--vae_channel", type=str, default="point", help="vae_channel")
    parser.add_argument("--if_color", type=int, default=1, help="")
    parser.add_argument("--if_mask", type=int, default=1, help="")
    parser.add_argument("--if_vr", type=int, default=0, help="")
    parser.add_argument("--if_special_corr", type=int, default=0, help="")
    parser.add_argument("--if_blank", type=int, default=0, help="")
    parser.add_argument("--resume_from_checkpoint", type=str, default="latest", help="resume_from_checkpoint")
    parser.add_argument("--enabled_deepspeed", default=False, action="store_true", help="Whether to use deepspeed.")
    parser.add_argument("--model_ds_config", type=str, default='deepspeed_config/zero_2.json', help="Path to a DeepSpeed config file.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", help="Mixed precision.")
    
    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    return parser



def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Paths to tokenizer.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    return parser
