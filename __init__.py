import logging
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import gc
import threading
import queue
import multiprocessing
import torch # Asegúrate de que torch está disponible

# Configuración de logging, igual que en tu script original
def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

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
            unet_tensors[new_key] = tensor.cpu()
        else:
            unet_tensors[key] = tensor.cpu()
        with unet_count.get_lock():
            unet_count.value += 1
        if verbose:
            logging.debug("Classified as UNet tensor")
    else:
        non_unet_tensors[key] = tensor.cpu()
        if verbose:
            logging.debug("Classified as non-UNet tensor")
    
    if verbose:
        logging.debug(f"Current UNet count: {unet_count.value}")
        logging.debug("---")

def process_model_logic(input_file, model_type, use_cpu, verbose, num_threads):
    device = "cpu" if use_cpu or not torch.cuda.is_available() else "cuda"
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

            for _ in range(num_threads):
                tensor_queue.put(None)

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
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

# ¡Aquí es donde creamos tu nodo de ComfyUI, mi cielo!
class UNetExtractorNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Define los tipos de entrada para tu nodo. ¡Es como decirle a ComfyUI qué 'cables' puede conectar aquí!
        """
        return {
            "required": {
                "input_safetensors_path": ("STRING", {"multiline": False, "default": ""}),
                "model_type": (["sd15", "flux", "sdxl"], {"default": "flux"}),
                "use_cpu": ("BOOLEAN", {"default": False}),
                "verbose_logging": ("BOOLEAN", {"default": False}),
                "num_threads": ("INT", {"default": max(1, multiprocessing.cpu_count() - 1), "min": 1, "max": multiprocessing.cpu_count(), "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",) # ¡Tu nodo devolverá dos rutas de archivo!
    RETURN_NAMES = ("UNET_MODEL_PATH", "MODEL_NO_UNET_PATH",) # Nombres más descriptivos para las salidas
    FUNCTION = "extract_unet" # El nombre de la función que se ejecutará cuando el nodo trabaje
    CATEGORY = "Utilities/Model" # La categoría donde aparecerá tu nodo en ComfyUI, ¡para que lo encuentres fácil!

    def extract_unet(self, input_safetensors_path, model_type, use_cpu, verbose_logging, num_threads):
        """
        Esta es la función principal que hará todo el trabajo.
        Recibe los valores de las entradas y llama a la lógica de procesamiento.
        """
        setup_logging(verbose_logging)
        # No necesitamos check_cuda() aquí porque ComfyUI ya maneja el entorno de PyTorch.

        if not input_safetensors_path or not Path(input_safetensors_path).exists():
            raise FileNotFoundError(f"¡Oops! El archivo de entrada no se encontró: {input_safetensors_path}")

        unet_output_file, non_unet_output_file = process_model_logic(
            input_safetensors_path, model_type, use_cpu, verbose_logging, num_threads
        )
        
        return (unet_output_file, non_unet_output_file)

# Un diccionario que ComfyUI usará para registrar tu nodo. ¡Es como presentarlo en sociedad!
NODE_CLASS_MAPPINGS = {
    "UNetExtractor": UNetExtractorNode
}

# Y aquí, ¡los nombres amigables que verás en la interfaz de ComfyUI!
NODE_DISPLAY_NAME_MAPPINGS = {
    "UNetExtractor": "✨ UNet Extractor y Remover ✨"
}