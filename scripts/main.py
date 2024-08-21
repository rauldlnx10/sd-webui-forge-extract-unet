import logging
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import gc
import threading
import queue
import multiprocessing
import gradio as gr
import os
import modules.scripts as scripts
from modules import script_callbacks

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "Unet Extractor"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return ()

    def on_ui_tabs(self):
        return [(self.interface, "UNet Extractor", "unet")]


try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def check_cuda():
    if CUDA_AVAILABLE:
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU device name: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("CUDA is not available. Using CPU.")

def is_unet_tensor(key, model_type):
    if model_type == "sd15":
        return key.startswith("model.diffusion_model.")
    elif model_type == "flux":
        return any(key.startswith(prefix) for prefix in [
            "unet.", "diffusion_model.", "model.diffusion_model.",
            "double_blocks.", "single_blocks.", "final_layer.",
            "guidance_in.", "img_in."
        ])
    elif model_type == "sdxl":
        return key.startswith("model.diffusion_model.")
    return False

def process_tensor(key, tensor, model_type, unet_tensors, non_unet_tensors, unet_count, verbose):
    if is_unet_tensor(key, model_type):
        if model_type == "sd15":
            new_key = key.replace("model.diffusion_model.", "")
            unet_tensors[new_key] = tensor.cpu()  # Move to CPU
        else:
            unet_tensors[key] = tensor.cpu()  # Move to CPU
        with unet_count.get_lock():
            unet_count.value += 1
        if verbose:
            logging.debug("Classified as UNet tensor")
    else:
        non_unet_tensors[key] = tensor.cpu()  # Move to CPU
        if verbose:
            logging.debug("Classified as non-UNet tensor")
    
    if verbose:
        logging.debug(f"Current UNet count: {unet_count.value}")
        logging.debug("---")

def process_model(input_file, model_type, use_cpu, verbose, num_threads):
    device = "cpu" if use_cpu or not CUDA_AVAILABLE else "cuda"
    logging.info(f"Processing {input_file} on {device}")
    logging.info(f"Model type: {model_type}")
    logging.info(f"Using {num_threads} threads")
    
    try:
        input_path = Path(input_file)
        base_name = input_path.stem
        output_dir = input_path.parent

        unet_output_file = output_dir / f"{base_name}_UNET.safetensors"
        non_unet_output_file = output_dir / f"{base_name}_MODEL_NO_UNET.safetensors"

        with safe_open(input_file, framework="pt", device=device) as f:
            unet_tensors = {}
            non_unet_tensors = {}
            total_tensors = 0
            unet_count = multiprocessing.Value('i', 0)
            key_prefixes = set()

            tensor_queue = queue.Queue()

            def worker():
                while True:
                    item = tensor_queue.get()
                    if item is None:
                        break
                    key, tensor = item
                    process_tensor(key, tensor, model_type, unet_tensors, non_unet_tensors, unet_count, verbose)
                    tensor_queue.task_done()

            threads = []
            for _ in range(num_threads):
                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)

            for key in f.keys():
                total_tensors += 1
                tensor = f.get_tensor(key)
                key_prefix = key.split('.')[0]
                key_prefixes.add(key_prefix)
                
                if verbose:
                    logging.debug(f"Processing key: {key}")
                    logging.debug(f"Tensor shape: {tensor.shape}")
                
                tensor_queue.put((key, tensor))

            # Signal threads to exit
            for _ in range(num_threads):
                tensor_queue.put(None)

            # Wait for all threads to complete
            for t in threads:
                t.join()

            logging.info(f"Total tensors processed: {total_tensors}")
            logging.info(f"UNet tensors: {unet_count.value}")
            logging.info(f"Non-UNet tensors: {total_tensors - unet_count.value}")
            logging.info(f"Unique key prefixes found: {', '.join(sorted(key_prefixes))}")

        if unet_count.value == 0:
            logging.warning("No UNet tensors were identified. Please check if the model type is correct.")

        logging.info(f"Saving extracted UNet to {unet_output_file}")
        save_file(unet_tensors, unet_output_file)
        
        logging.info(f"Saving model without UNet to {non_unet_output_file}")
        save_file(non_unet_tensors, non_unet_output_file)
        
        logging.info("Processing complete!")
        return str(unet_output_file), str(non_unet_output_file)

    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")
        raise
    finally:
        # Clean up GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

def gradio_process(input_file, model_type, use_cpu, verbose, num_threads):
    try:
        setup_logging(verbose)
        check_cuda()

        if num_threads is None:
            num_threads = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free

        unet_output_file, non_unet_output_file = process_model(input_file, model_type, use_cpu, verbose, num_threads)
        return f"UNet output file: {unet_output_file}\nModel without UNet output file: {non_unet_output_file}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

with gr.Blocks() as interface:
    with gr.Row():
        gr.Markdown("# UNet Extractor and Remover for Stable Diffusion 1.5, SDXL, and FLUX")
        gr.Markdown("[Github](https://github.com/captainzero93/extract-unet-safetensor)")

    with gr.Row():
        with gr.Group():
            input_file = gr.Textbox(label="Input SafeTensors File Path", placeholder="Enter the file path")
            model_type = gr.Radio(label="Model Type", choices=["sd15", "flux", "sdxl"], type="value")
            use_cpu = gr.Checkbox(label="Use CPU", value=False)
            verbose = gr.Checkbox(label="Verbose Logging", value=False)
            num_threads = gr.Slider(label="Number of Threads", minimum=1, maximum=multiprocessing.cpu_count(), step=1, value=multiprocessing.cpu_count()-1)
                    
    with gr.Row():
        with gr.Group():
            output_text = gr.Textbox(label="Output", placeholder="Processing results will appear here...", visible=True)
            process_btn = gr.Button("Process Model", variant="primary")

        def on_process_click(file_path, model, cpu, verbose, threads):
            return gradio_process(file_path, model, cpu, verbose, threads)
        
        process_btn.click(on_process_click, inputs=[input_file, model_type, use_cpu, verbose, num_threads], outputs=output_text)

    Script.interface = interface

script = Script()
script_callbacks.on_ui_tabs(script.on_ui_tabs)
