import gradio as gr
import numpy as np
import cv2
import torch
import os
from PIL import Image

# ==========================================
# 1. SAM Model Wrapper
# ==========================================
class SAMWrapper:
    def __init__(self, checkpoint_path="sam_vit_h_4b8939.pth", model_type="vit_h"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = None
        
        if os.path.exists(checkpoint_path):
            print(f"Loading SAM model ({self.device})...")
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            print("SAM model loaded.")
        else:
            print("Warning: SAM checkpoint not found. Running in 'Mock Mode'.")

    def set_image(self, image):
        if self.predictor:
            self.predictor.set_image(image)

    def predict(self, points, labels, image_shape):
        if self.predictor:
            masks, _, _ = self.predictor.predict(
                point_coords=np.array(points),
                point_labels=np.array(labels),
                multimask_output=False,
            )
            return masks[0]
        else:
            h, w = image_shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            if len(points) > 0:
                cx, cy = points[-1]
                size = 100
                x1 = max(0, int(cx - size/2))
                y1 = max(0, int(cy - size/2))
                x2 = min(w, int(cx + size/2))
                y2 = min(h, int(cy + size/2))
                mask[y1:y2, x1:x2] = True
            return mask

sam_model = SAMWrapper("./checkpoints/sam_vit_h_4b8939.pth")

# ==========================================
# 2. Image Processing and Dimension Analysis Logic
# ==========================================

def draw_overlay(image, mask, points, labels):
    if image is None: return None
    vis_img = image.copy()
    if mask is not None:
        color_mask = np.zeros_like(vis_img)
        color_mask[mask] = [0, 255, 0]
        vis_img = cv2.addWeighted(vis_img, 0.7, color_mask, 0.3, 0)
    for pt, lbl in zip(points, labels):
        color = (0, 255, 0) if lbl == 1 else (255, 0, 0)
        cv2.circle(vis_img, tuple(pt), 6, color, -1)
        cv2.circle(vis_img, tuple(pt), 6, (255, 255, 255), 1)
    return vis_img

def process_crop_logic(original_img, mask):
    if original_img is None or mask is None: return None
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0: return original_img
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    crop_h = y_max - y_min + 1
    crop_w = x_max - x_min + 1
    result_crop = np.full((crop_h, crop_w, 3), 128, dtype=np.uint8)
    original_crop = original_img[y_min:y_max+1, x_min:x_max+1]
    mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
    result_crop[mask_crop] = original_crop[mask_crop]
    return result_crop

def overlay_image_relative(background, overlay, x_ratio, y_ratio, scale):
    if background is None: return None
    if overlay is None: return background
    bg_h, bg_w = background.shape[:2]
    new_w = int(overlay.shape[1] * scale)
    new_h = int(overlay.shape[0] * scale)
    if new_w <= 0 or new_h <= 0: return background
    resized_overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x = int(x_ratio * bg_w)
    y = int(y_ratio * bg_h)
    result = background.copy()
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + new_w, bg_w), min(y + new_h, bg_h)
    if x1 >= x2 or y1 >= y2: return result
    overlay_x1, overlay_y1 = x1 - x, y1 - y
    overlay_x2, overlay_y2 = overlay_x1 + (x2 - x1), overlay_y1 + (y2 - y1)
    result[y1:y2, x1:x2] = resized_overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    return result

def generate_preview_cache(image, max_size=800):
    if image is None: return None, 1.0
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        small_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return small_img, scale
    return image.copy(), 1.0

def save_result_as_jpg(image):
    if image is None: return None
    try:
        pil_img = Image.fromarray(image)
        output_path = "final_result.jpg"
        pil_img.save(output_path, format="JPEG", quality=95)
        return output_path
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def check_dimensions(h, w):
    """
    Target is W 832 : H 480
    """
    if h == 0: return ""
    
    target_ratio = 832 / 480
    current_ratio = w / h
    
    tolerance = 0.05 
    diff = abs(current_ratio - target_ratio)
    
    dim_str = f"{w} (W) x {h} (H)"
    
    msg = f"### 📏 Dimensions: {dim_str}"
    
    if diff < tolerance:
        msg += f" <span style='color: green;'>✅ (Ratio ~832:480 Good)</span>"
    else:
        msg += f"<br><span style='color: red;'>⚠️ **Warning**: Distortion Risk! Target ratio is 832(W):480(H).</span>"
        
    return msg

# ==========================================
# 3. Gradio Interaction Logic
# ==========================================

def init_app():
    with gr.Blocks(title="Smart Image Composition") as demo:
        gr.Markdown("## 🎨 Smart Image Composition Workbench")
        
        # --- Global State ---
        state_current_bg = gr.State(None) 
        state_preview_bg = gr.State(None) 
        state_preview_scale = gr.State(1.0) 

        state_subject_raw = gr.State(None) 
        state_points = gr.State([]) 
        state_labels = gr.State([]) 
        state_current_mask = gr.State(None) 
        state_cropped_subject = gr.State(None) 
        
        with gr.Row():
            # Left: Current Working Canvas
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Canvas Preview")
                
                canvas_display = gr.Image(label="Result", interactive=False, format="jpeg")
                
                # Dimension Display
                dims_display = gr.Markdown("📏 Waiting for image...")
                
                with gr.Accordion("Canvas Expansion (Step 3)", open=False):
                    gr.Markdown("ℹ️ **Cumulative Mode**: Updates apply on confirm.")
                    with gr.Row():
                        pad_top = gr.Number(label="Top (px)", value=0, precision=0)
                        pad_bottom = gr.Number(label="Bottom (px)", value=0, precision=0)
                    with gr.Row():
                        pad_left = gr.Number(label="Left (px)", value=0, precision=0)
                        pad_right = gr.Number(label="Right (px)", value=0, precision=0)
                    
                    btn_confirm_expand = gr.Button("✅ Confirm & Expand", variant="primary")

            # Right: Operations Panel
            with gr.Column(scale=1):
                gr.Markdown("### 🛠️ Operations")
                
                # --- Step 1: Initialization ---
                with gr.Tab("1. Init Background"):
                    init_bg_input = gr.Image(label="Upload Background", type="numpy")
                    btn_init_bg = gr.Button("Set as Canvas", variant="primary")

                # --- Step 2: Add Subject ---
                with gr.Tab("2. Add Subject"):
                    gr.Markdown("**Step 2.1: Segment Subject**")
                    subject_input = gr.Image(label="Subject Image", type="numpy")
                    radio_point_type = gr.Radio(["Keep (+)", "Remove (-)"], value="Keep (+)", label="Point Type")
                    btn_confirm_crop = gr.Button("Confirm Crop", variant="primary")
                    
                    gr.Markdown("---")
                    gr.Markdown("**Step 2.2: Position & Paste**")
                    cropped_preview = gr.Image(label="Crop Result", interactive=False, height=150)
                    
                    slider_x = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="X Ratio")
                    slider_y = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Y Ratio")
                    slider_scale = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Scale")
                    
                    btn_confirm_paste = gr.Button("✅ Confirm Paste", variant="primary")
                
                # --- Step 3: Export ---
                gr.Markdown("### 💾 Export")
                with gr.Group():
                    btn_generate_jpg = gr.Button("Generate JPG Link")
                    file_download = gr.File(label="Download File", file_count="single")

        # ================= Logic Implementation =================

        def update_all_bg_states(img):
            if img is None: return [None]*3
            small, scale = generate_preview_cache(img)
            return [img, small, scale]

        # 1. Initialize Background
        def on_init_bg(img):
            if img is None: return [None]*5
            h, w = img.shape[:2]
            msg = check_dimensions(h, w)
            states = update_all_bg_states(img) 
            return states + [img, msg]
            
        btn_init_bg.click(on_init_bg, inputs=[init_bg_input], 
                          outputs=[state_current_bg, state_preview_bg, state_preview_scale, canvas_display, dims_display])
        
        # --- Expansion Logic ---

        # A. Fast Preview + Dimension Prediction
        def fast_preview_expansion(full_bg, small_bg, scale_factor, t, b, l, r):
            if full_bg is None: return None, ""
            
            if t==0 and b==0 and l==0 and r==0:
                h, w = full_bg.shape[:2]
                return full_bg, check_dimensions(h, w)
            
            if small_bg is None: return full_bg, ""

            # 1. Generate Low-Resolution Preview
            t_s, b_s = int(t * scale_factor), int(b * scale_factor)
            l_s, r_s = int(l * scale_factor), int(r * scale_factor)
            padded_small = np.pad(small_bg, ((t_s, b_s), (l_s, r_s), (0, 0)), mode='constant', constant_values=128)
            
            # 2. Calculate Predicted High-Res Dimensions (Base + Increment)
            base_h, base_w = full_bg.shape[:2]
            new_h = base_h + int(t) + int(b)
            new_w = base_w + int(l) + int(r)
            dim_msg = check_dimensions(new_h, new_w)
            
            return padded_small, dim_msg

        # B. Confirm Expansion
        def confirm_expansion(current_bg, t, b, l, r):
            if current_bg is None: return [None]*8
            
            final_bg = np.pad(current_bg, ((int(t), int(b)), (int(l), int(r)), (0, 0)), mode='constant', constant_values=128)
            states = update_all_bg_states(final_bg)
            
            # Calculate final dimension text
            h, w = final_bg.shape[:2]
            msg = check_dimensions(h, w)
            
            return states + [final_bg, msg, 0, 0, 0, 0]

        # Bind Events
        expand_inputs = [state_current_bg, state_preview_bg, state_preview_scale, pad_top, pad_bottom, pad_left, pad_right]
        
        pad_top.change(fast_preview_expansion, inputs=expand_inputs, outputs=[canvas_display, dims_display])
        pad_bottom.change(fast_preview_expansion, inputs=expand_inputs, outputs=[canvas_display, dims_display])
        pad_left.change(fast_preview_expansion, inputs=expand_inputs, outputs=[canvas_display, dims_display])
        pad_right.change(fast_preview_expansion, inputs=expand_inputs, outputs=[canvas_display, dims_display])
        
        btn_confirm_expand.click(
            confirm_expansion,
            inputs=[state_current_bg, pad_top, pad_bottom, pad_left, pad_right],
            outputs=[state_current_bg, state_preview_bg, state_preview_scale, canvas_display, dims_display,
                     pad_top, pad_bottom, pad_left, pad_right]
        )

        # --- Subject Segmentation ---
        def on_subject_upload(img):
            if img is not None: sam_model.set_image(img)
            return img, img, [], [], None
        subject_input.upload(on_subject_upload, inputs=[subject_input], outputs=[state_subject_raw, subject_input, state_points, state_labels, state_current_mask])

        def on_click_update(raw_img, evt: gr.SelectData, points, labels, mode):
            if raw_img is None: return raw_img, points, labels, None
            x, y = evt.index[0], evt.index[1]
            label = 1 if "Keep" in mode else 0
            points.append([x, y])
            labels.append(label)
            mask = sam_model.predict(points, labels, raw_img.shape)
            vis_img = draw_overlay(raw_img, mask, points, labels)
            return vis_img, points, labels, mask

        subject_input.select(on_click_update, inputs=[state_subject_raw, state_points, state_labels, radio_point_type], outputs=[subject_input, state_points, state_labels, state_current_mask])

        def do_crop(img, mask):
            if img is None or mask is None: return None, None
            cropped = process_crop_logic(img, mask)
            return cropped, cropped
        btn_confirm_crop.click(do_crop, inputs=[state_subject_raw, state_current_mask], outputs=[state_cropped_subject, cropped_preview])

        # --- Pasting Logic ---
        def fast_preview_paste_wrapper(small_bg, preview_scale, overlay, x_r, y_r, s):
            if small_bg is None or overlay is None: return None
            if preview_scale < 1.0:
                h, w = overlay.shape[:2]
                p_w, p_h = int(w * preview_scale), int(h * preview_scale)
                overlay_small = cv2.resize(overlay, (p_w, p_h)) if p_w > 0 else overlay
            else:
                overlay_small = overlay
            return overlay_image_relative(small_bg, overlay_small, x_r, y_r, s)

        paste_preview_inputs = [state_preview_bg, state_preview_scale, state_cropped_subject, slider_x, slider_y, slider_scale]
        
        slider_x.change(fast_preview_paste_wrapper, inputs=paste_preview_inputs, outputs=canvas_display)
        slider_y.change(fast_preview_paste_wrapper, inputs=paste_preview_inputs, outputs=canvas_display)
        slider_scale.change(fast_preview_paste_wrapper, inputs=paste_preview_inputs, outputs=canvas_display)

        def confirm_paste(current_bg, overlay, x_r, y_r, s):
            if current_bg is None: return [None]*5
            final_bg = overlay_image_relative(current_bg, overlay, x_r, y_r, s)
            states = update_all_bg_states(final_bg)
            
            h, w = final_bg.shape[:2]
            msg = check_dimensions(h, w)
            return states + [final_bg, msg]

        btn_confirm_paste.click(
            confirm_paste, 
            inputs=[state_current_bg, state_cropped_subject, slider_x, slider_y, slider_scale], 
            outputs=[state_current_bg, state_preview_bg, state_preview_scale, canvas_display, dims_display]
        )

        # --- Export ---
        btn_generate_jpg.click(
            save_result_as_jpg, 
            inputs=[state_current_bg], 
            outputs=[file_download]
        )

    return demo

if __name__ == "__main__":
    app = init_app()
    app.launch()