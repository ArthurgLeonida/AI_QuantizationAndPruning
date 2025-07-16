import torch
import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer # Keep AutoConfig for config loading
import glob
import time

def quantize_PTQ_model(model_path: str, quantized_model_save_path: str):
    """
    Applies dynamic quantization to a fine-tuned Question Answering model.
    This quantizes weights from FP32 to INT8 at inference time.

    Args:
        model_path (str): Path to the directory containing the full-precision fine-tuned model.
        quantized_model_save_path (str): Path to save the dynamically quantized model.
    """
    if not os.path.isdir(model_path):
        print(f"Error: Full-precision model not found at {model_path}.")
        return None

    print(f"\nLoading full-precision model from: {model_path} for quantization...")
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.eval() # Set model to evaluation mode

    print("Applying dynamic quantization (FP32 -> INT8) to the model...")
    quantized_model_obj = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    print("Dynamic quantization applied.")

    if not os.path.exists(quantized_model_save_path):
        os.makedirs(quantized_model_save_path)

    # --- FIX: Save the entire quantized model object directly ---
    # This is the recommended way to save dynamically quantized models in PyTorch.
    # It saves the model's structure and state.
    torch.save(quantized_model_obj, os.path.join(quantized_model_save_path, "quantized_model.pth"))
    print(f"Quantized model object saved to: {os.path.join(quantized_model_save_path, 'quantized_model.pth')}")
    # --- END FIX ---

    # Save model config (from original model) and tokenizer (from original model path)
    # These are still needed even if loading the model object directly, for consistency
    # and for AutoTokenizer.from_pretrained to work.
    model.config.save_pretrained(quantized_model_save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path) 
    tokenizer.save_pretrained(quantized_model_save_path)
    print("Quantized model config and tokenizer saved.")
    
    return quantized_model_obj

def measure_model_size(model_path: str):

    """
    Measures the size of a saved model in Megabytes, looking for common weight file names.
    This function is adapted to handle both 'model.safetensors' and 'pytorch_model.bin'.

    Args:
        model_path (str): Path to the directory containing the saved model weights.

    Returns:
        float: The size of the model file in Megabytes, or 0.0 if not found/error.
    """
    if not os.path.isdir(model_path):
        print(f"Error: Model directory not found at '{model_path}'.")
        return 0.0
    
    # --- Check for common weight file names in preferred order ---
    model_file_name = None
    
    # Prefer .safetensors if it exists (newer format)
    if os.path.exists(os.path.join(model_path, "model.safetensors")):
        model_file_name = "model.safetensors"
    # Fallback to .bin if .safetensors is not found
    elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        model_file_name = "pytorch_model.bin"
    else:
        # Fallback for other common weight files, if specific names are not present
        # This globbing handles cases where the model might be saved with a different name
        # but still uses .safetensors or .bin extensions.
        safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        pytorch_bin_files = glob.glob(os.path.join(model_path, "*.bin"))
        
        if len(safetensors_files) > 0:
            model_file_name = os.path.basename(safetensors_files[0]) # Take the first one found
        elif len(pytorch_bin_files) > 0:
            model_file_name = os.path.basename(pytorch_bin_files[0]) # Take the first one found
        else:
            print(f"Error: No recognized model weight file (.safetensors or .bin) found in '{model_path}'.")
            return 0.0

    actual_model_file_path = os.path.join(model_path, model_file_name)

    if os.path.exists(actual_model_file_path):
        size_bytes = os.path.getsize(actual_model_file_path)
        size_mb = size_bytes / (1024 * 1024)
        print(f"Model size at '{model_path}' ({model_file_name}): {size_mb:.2f} MB")
        return size_mb
    else:
        # This case should ideally not be hit if the prior checks are exhaustive
        print(f"Error: Model file '{actual_model_file_path}' not found after selection logic.")
        return 0.0
    
def benchmark_inference_speed(
    model_path: str,
    tokenizer_path: str, # Need tokenizer for vocab_size and sep_token_id
    is_quantized: bool = False,
    device_str: str = "cpu", # Use "cpu" or "cuda"
    batch_size: int = 16,
    sequence_length: int = 512,
    num_warmup_runs: int = 10,
    num_timed_runs: int = 100,
):
    """
    Benchmarks the pure inference speed of a model on a specified device.

    Args:
        model_path (str): Path to the saved model.
        tokenizer_path (str): Path to the tokenizer (for model config like vocab_size).
        is_quantized (bool): True if loading a quantized model.
        device_str (str): Target device for inference: "cpu" or "cuda".
        batch_size (int): Batch size for inference.
        sequence_length (int): Max sequence length for dummy input.
        num_warmup_runs (int): Number of runs to warm up the GPU/CPU.
        num_timed_runs (int): Number of runs to time.

    Returns:
        float: Samples per second (inferences * batch_size / total_time).
    """
    if not os.path.isdir(model_path) or not os.path.isdir(tokenizer_path):
        print(f"Error: Model or tokenizer directory not found at {model_path} or {tokenizer_path}.")
        return 0.0

    print(f"\nBenchmarking '{os.path.basename(model_path)}' on {device_str.upper()}...")
    
    # Load tokenizer to get vocab_size (needed for dummy input) and potentially for config
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Load model (logic needs to differentiate quantized vs non-quantized)
    if is_quantized:
        try:
            # Load the entire quantized model object directly from the .pth file.
            model = torch.load(os.path.join(model_path, "quantized_model.pth"), map_location=torch.device("cpu"), weights_only=False)
            model.eval() # Set model to evaluation mode
            model.to(torch.device("cpu")) # Quantized model MUST stay on CPU
            if device_str == "cuda": # Warn if user tries to benchmark quantized on CUDA
                print("WARNING: Quantized dynamic model cannot run on CUDA backend with this PyTorch build. Benchmarking will proceed on CPU.")
                device_str = "cpu" # Force CPU for correct result
        except Exception as e:
            print(f"Error loading quantized model for benchmarking from {model_path}: {e}")
            return 0.0
    else:
        # Load full-precision model
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        model.eval() # Set model to evaluation mode
        # Move to GPU if requested and available, else CPU
        if device_str == "cuda" and torch.cuda.is_available():
            model.to(torch.device("cuda"))
        else:
            model.to(torch.device("cpu"))

    # Generate dummy input tensors on the model's actual device
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, sequence_length), device=model.device)
    dummy_attention_mask = torch.ones(batch_size, sequence_length, device=model.device)
    
    # Warm-up runs (to prime GPU, caches, etc.)
    with torch.no_grad():
        for _ in range(num_warmup_runs):
            _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
        # Synchronize GPU for accurate timing after warm-up
        if model.device.type == 'cuda':
            torch.cuda.synchronize()

    # Timed runs
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_timed_runs):
            _ = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
        # Synchronize GPU again before stopping timer
        if model.device.type == 'cuda':
            torch.cuda.synchronize()
    end_time = time.time()

    total_time_seconds = end_time - start_time
    samples_per_second = (batch_size * num_timed_runs) / total_time_seconds
    
    print(f"Benchmarking complete. Samples per second on {model.device.type.upper()}: {samples_per_second:.2f}")
    return samples_per_second