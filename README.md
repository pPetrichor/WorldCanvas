# The World is Your Canvas: Painting Promptable Events with Reference Images, Trajectories, and Text


**[**[**📄 Paper**](https://arxiv.org/abs/2512.16924)**]**
**[**[**🌐 Project Page**](https://worldcanvas.github.io/)**]**
**[**[**🤗 Model Weights**](https://huggingface.co/hlwang06/WorldCanvas)**]**

https://github.com/user-attachments/assets/cc8f7fd6-fd89-47e9-b2bf-38298131d1f7


_**[Hanlin Wang<sup>1,2</sup>](https://scholar.google.com/citations?user=0uO4fzkAAAAJ&hl=zh-CN), [Hao Ouyang<sup>2</sup>](https://ken-ouyang.github.io/), [Qiuyu Wang<sup>2</sup>](https://github.com/qiuyu96), [Yue Yu<sup>1,2</sup>](https://bruceyy.com/), [Yihao Meng<sup>1,2</sup>](https://yihao-meng.github.io/), <br> [Wen Wang<sup>3,2</sup>](https://github.com/encounter1997), [Ka Leong Cheng<sup>2</sup>](https://felixcheng97.github.io/), [Shuailei Ma<sup>4,2</sup>](https://scholar.google.com/citations?user=dNhzCu4AAAAJ&hl=zh-CN), [Qingyan Bai<sup>1,2</sup>](https://bqy.info/), [Yixuan Li<sup>5,2</sup>](https://yixuanli98.github.io/), <br> [Cheng Chen<sup>6,2</sup>](https://scholar.google.com/citations?user=nNQU71kAAAAJ&hl=zh-CN), [Yanhong Zeng<sup>2</sup>](https://zengyh1900.github.io/), [Xing Zhu<sup>2</sup>](https://openreview.net/profile?id=~Xing_Zhu2), [Yujun Shen<sup>2</sup>](https://shenyujun.github.io/), [Qifeng Chen<sup>1</sup>](https://cqf.io/)**_
<br>
<sup>1</sup>HKUST, <sup>2</sup>Ant Group, <sup>3</sup>ZJU, <sup>4</sup>NEU, <sup>5</sup>CUHK, <sup>6</sup>NTU

# TLDR
WorldCanvas is a I2V framework for promptable world events that enables rich, user-directed simulation by combining text, trajectories, and reference images.

**Strongly recommend seeing our [demo page](https://worldcanvas.github.io/).**

If you enjoyed the videos we created, please consider giving us a star 🌟.

## 🚀 Open-Source Plan

### ✅ Released
*   Full inference code
*   `WorldCanvas-14B` 
*   `WorldCanvas-14B-ref`



# Setup
```shell
git clone https://github.com/pPetrichor/WorldCanvas.git
cd WorldCanvas
```
# Environment
We use a environment similar to diffsynth. If you have a diffsynth environment, you can probably reuse it. Our environment also requires [SAM](https://github.com/facebookresearch/segment-anything) to be installed.
```shell
conda create -n WorldCanvas python=3.10
conda activate WorldCanvas
pip install -e .
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

We use FlashAttention-3 to implement the sparse inter-shot attention. We highly recommend using FlashAttention-3 for its fast speed. We provide a simple instruction on how to install FlashAttention-3.

```shell
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
cd hopper
python setup.py install
```
If you encounter environment problem when installing FlashAttention-3, you can refer to their official github page https://github.com/Dao-AILab/flash-attention.

If you cannot install FlashAttention-3, you can use FlashAttention-2 as an alternative, and our code will automatically detect the FlashAttention version. It will be slower than FlashAttention-3,but can also produce the right result.

If you want to install FlashAttention-2, you can use the following command:
```shell
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

# Checkpoint


### Step 1: Download Wan 2.2 VAE and T5
If you already have downloaded Wan 2.2 14B T2V before, skip this section.

If not, you need the T5 text encoder and the VAE from the original Wan 2.2 repository:
[https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)


Based on the repository's file structure, you **only** need to download `models_t5_umt5-xxl-enc-bf16.pth` and `Wan2.1_VAE.pth`.

You do **not** need to download the `google`, `high_noise_model`, or `low_noise_model` folders, nor any other files. 

#### Recommended Download (CLI)

We recommend using `huggingface-cli` to download only the necessary files. Make sure you have `huggingface_hub` installed (`pip install huggingface_hub`).

This command will download *only* the required T5 and VAE models into the correct directory:

```bash
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B \
  --local-dir checkpoints/Wan2.2-T2V-A14B \
  --allow-patterns "models_t5_*.pth" "Wan2.1_VAE.pth"
```

#### Manual Download

Alternatively, go to the "Files" tab on the Hugging Face repo and manually download the following two files:

  * `models_t5_umt5-xxl-enc-bf16.pth`
  * `Wan2.1_VAE.pth`

Place both files inside a new folder named `checkpoints/Wan2.2-T2V-A14B/`.

### Step 2: Download SAM Model

Download SAM vit_h model from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

### Step 3: Download WorldCanvas Model (WorldCanvas\_dit)

Download our fine-tuned high-noise and low-noise DiT checkpoints from the following link:

**[➡️ Download WorldCanvas\_dit Model Checkpoints [Here](https://huggingface.co/hlwang06/WorldCanvas)]**

This download contain the four fine-tuned model files. Two for no reference images version: `WorldCanvas/high_model.safetensors`, `WorldCanvas/low_model.safetensors`. And two for reference-based version: `WorldCanvas_ref/high_model.safetensors`, `WorldCanvas_ref/low_model.safetensors`.

### Step 4: Final Directory Structure

Make sure your `checkpoints` directory look like this:

```
checkpoints/
├── sam_vit_h_4b8939.pth
├── Wan2.2-T2V-A14B/
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   └── Wan2.1_VAE.pth
└── WorldCanvas_dit/
    ├── WorldCanvas/
    │    ├── high_model.safetensors
    │    └── low_model.safetensors
    └── WorldCanvas_ref/
         ├── high_model.safetensors
         └── low_model.safetensors
```


# Inference without reference image
If you don't have a reference image, you can proceed with inference as follows:

### 1. Generate your generation conditions with gradio

```shell
cd gradio
python draw_traj.py
```
#### (a) In the opened interface, enter the path to the initial image and click "Load Image."

<img src="pics/traj1.png" alt="SVG image" width="800">

#### (b) Then use SAM to select the subject you want to manipulate. You can change the type in "Point Type" to determine the type of SAM point to add. After making your selection, click "Confirm Mask."

<img src="pics/traj2.png" alt="SVG image" width="800">

#### (c)  Now, draw a trajectory for your selected subject. By clicking directly on the image, you can create a path that sequentially connects these points. The time interval between each pair of consecutive points will be treated as equal, meaning that the smaller the distance between points, the slower the movement speed; conversely, the larger the distance, the faster the movement speed. In the illustration, we clicked three points to form the trajectory.

<img src="pics/traj3.png" alt="SVG image" width="600">

#### After finishing the clicking, set your desired start and end times for the trajectory in the "st" and "et" fields under 'Stage Two: Trajectory Drawing' (this can simulate the object suddenly appearing or disappearing). Once confirmed, click "Generate Trajectory [st, et)" to create the drawn trajectory.

#### (d) Finally, in the "Stage Three" panel, fill in the "Object ID" and its corresponding text in "Text Description". Click "Confirm and Add to Results" to record all the conditions. (Note: A single subject performing the same action can have multiple trajectories, but they should all share the same Object ID and text.)

<img src="pics/traj4.png" alt="SVG image" width="800">

#### (e) Next, you can repeat the above steps to obtain multiple control conditions.

Tip 1: After performing step (c), you can also erase certain segments of the generated trajectory to indicate that those parts are invisible, simulating scenarios such as occlusion or rotation. Simply click "Erase Mode" after clicking "Generate Trajectory [st, et)", then select any two points along the trajectory—the segment between these two points will be marked as invisible and displayed in red. Remember to turn off Erase Mode once you've finished erasing.

For example, to achieve a rotation effect for a puppy, you can perform the following steps:

<img src="pics/traj6.png" alt="SVG image" width="800">

<img src="pics/traj7.png" alt="SVG image" width="800">

Tip 2: To simulate the effect of a new object appearing out of nowhere, you can first select any arbitrary mask:

<img src="pics/traj8.png" alt="SVG image" width="800">

Then, when generating the trajectory, simply set the trajectory's start time to the moment you want the object to appear.

<img src="pics/traj9.png" alt="SVG image" width="800">

Also, associate the corresponding text with this trajectory.

<img src="pics/traj10.png" alt="SVG image" width="800">

Similarly, to simulate the effect of an object disappearing, simply set the trajectory's end time to the desired moment when the object should vanish, and then assign the corresponding text.

Tip 3: If you want to keep the camera stationary, you can randomly select several background objects. When drawing the trajectory, click only a single point for each mask and set the trajectory time range to 0–81 to create static trajectories. For these trajectories, simply enter "None" as the corresponding text; the code will automatically ignore this text and consider only the trajectory for camera control purposes.

If you want to control camera motion, you can randomly select several background objects and draw movement trajectories for them. Similarly, just enter "None" as the corresponding text for these trajectories.

#### (f) Finally, set the save path and save the generated conditions to a JSON file by clicking "Save JSON".

<img src="pics/traj11.png" alt="SVG image" width="800">

### 2. Generate video with your generation conditions
Set the 'sample' variable on line 174 of `WorldCanvas_inference.py` to ['your initial image path', 'your JSON file path'], and set the save paths on lines 247 and 248, then run:
```bash
python WorldCanvas_inference.py --seed 0
```
You can change the seed to obtain different results.

# Inference with reference images
If you have reference images, you first need to generate an image with the reference image added with gradio:
```shell
cd gradio
python ref_image_generation.py
```
You will see an interface like this:

<img src="pics/ref1.png" alt="SVG image" width="800">

First, upload the background image.

<img src="pics/ref2.png" alt="SVG image" width="800">

Then, if you need to expand the canvas, click "Canvas Expansion" below and set the number of pixels to extend in the top, bottom, left, and right directions. After confirming your settings, click "Confirm & Expand".

<img src="pics/ref3.png" alt="SVG image" width="800">

Next, click "2. Add Subject" in the top-right corner to add the reference image. You need to upload the reference image and use the SAM model to select the content you want. Click "Confirm Crop" to confirm. (Note: When you first enter "2. Add Subject," the background image will temporarily disappear—this is normal and can be ignored. Simply continue with the process, and the background will reappear automatically in "Step 2.2.")

<img src="pics/ref4.png" alt="SVG image" width="300">

Next, adjust the parameters in "Step 2.2" to control the size and position of the reference image. After confirming the reference image mask in 'Step 2.1', the reference image will not immediately appear on the background. It will only be displayed in real time on the background when you adjust the parameters in this step. Note that during adjustment, the preview will appear very blurry, but once you've finalized the size and position and click "Confirm Paste," the result will become clear.

<img src="pics/ref5.png" alt="SVG image" width="800">

You can repeat the above steps to continuously adjust the canvas size and insert any number of reference images. (Click the ❌ in the top-right corner of the reference image block in 'Step 2.1' to delete the current image and add a new one.)

Finally, click "Generate JPG Link" to download the resulting image. (Note: Before saving the result, please check the "Dimensions" hint below the image and try to keep the aspect ratio as close as possible to 832(width):480(height). This is because we will resize the initial image when drawing trajectories, and a significant deviation from this ratio may cause distortion of the main subject.)

Once you have the resulting composite image, you can follow steps in [Inference without reference image](#inference-without-reference-image) and use this composite image as the initial image to draw the conditions for video generation.

After generating conditions, set the 'sample' variable on line 174 of `WorldCanvas_inference_refimage.py` to ['your initial image path', 'your JSON file path'], and set the save paths on lines 247 and 248, then run:
```bash
python WorldCanvas_inference_refimage.py --seed 0
```
You can change the seed to obtain different results.

## Examples

We provide several examples directly within the `example` folder and use these examples in `WorldCanvas_inference.py` and `WorldCanvas_inference_refimage.py` script. You can try them out immediately.


# Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{wang2025worldcanvas,
  title={The World is Your Canvas: Painting Promptable Events with Reference Images, Trajectories, and Text},
  author={Hanlin Wang and Hao Ouyang and Qiuyu Wang and Yue Yu and Yihao Meng and Wen Wang and Ka Leong Cheng and Shuailei Ma and Qingyan Bai and Yixuan Li and Cheng Chen and Yanhong Zeng and Xing Zhu and Yujun Shen and Qifeng Chen},
  journal={arXiv preprint arXiv:2512.16924},
  year={2025}
}
```

# License

This project is licensed under the CC BY-NC-SA 4.0 ([Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/)).

The code is provided for academic research purposes only.

For any questions, please contact ymengas@connect.ust.hk.

