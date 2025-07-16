import torch
import os
import time
import torch.nn.utils.prune as prune
import random
import glob
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from tqdm.auto import tqdm


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

def quantize_PTSQ_model(
    model_path: str, 
    quantized_model_save_path: str,
    calibration_dataset, # A Dataset for calibration (a small subset of training data)
    tokenizer # Tokenizer needed for calibration data processing in eval_loop
):
    """
    Applies Post-Training Static Quantization (PTSQ) to a fine-tuned QA model.
    This quantizes both weights and activations to INT8 using a calibration dataset.

    Args:
        model_path (str): Path to the directory containing the full-precision fine-tuned model.
        quantized_model_save_path (str): Path to save the statically quantized model.
        calibration_dataset (datasets.Dataset): A small subset of representative data for calibration.
        tokenizer: The tokenizer object (needed for data processing in evaluation loop).
    """
    if not os.path.isdir(model_path):
        print(f"Error: Full-precision model not found at {model_path}.")
        return None

    print(f"\nLoading full-precision model from: {model_path} for static quantization...")
    # Load model (make sure it's on CPU before quantization for static)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.eval() # Set model to evaluation mode (crucial for quantization)
    model.to(torch.device("cpu")) # Ensure model is on CPU for static quantization

    # --- 1. Prepare the model for static quantization ---
    # This inserts observers that collect statistics (e.g., min/max ranges for activations)
    print("Preparing model for static quantization (inserting observers)...")
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm') # 'fbgemm' for server CPUs, 'qnnpack' for mobile CPUs
    # You might need to fuse modules for better performance (e.g., Conv-ReLU, Linear-ReLU)
    # For Transformers, fusing is less common automatically but can be done manually.
    # For now, we rely on default observers.
    torch.quantization.prepare(model, inplace=True)
    print("Model prepared for static quantization.")

    # --- 2. Calibrate the model ---
    # Run a few batches of representative data through the model to collect statistics.
    print(f"Calibrating model with {len(calibration_dataset)} examples...")
    
    # Create a DataLoader for calibration
    # Ensure calibration_dataset provides input_ids and attention_mask
    calibration_loader = torch.utils.data.DataLoader(
        calibration_dataset, 
        batch_size=16, # Use a typical batch size for calibration
        shuffle=False, # Order doesn't matter for calibration
        num_workers=0 # No multiprocessing for calibration to avoid complexity
    )

    with torch.no_grad():
        for i, batch in enumerate(tqdm(calibration_loader, desc="Calibrating")):
            # Move batch to CPU (model is on CPU)
            input_ids = batch['input_ids'].to(torch.device("cpu"))
            attention_mask = batch['attention_mask'].to(torch.device("cpu"))
            model(input_ids=input_ids, attention_mask=attention_mask) # Forward pass only

    print("Model calibration complete.")

    # --- 3. Convert the model to a quantized version ---
    # This uses the collected statistics to quantize weights and insert de/quantize ops.
    print("Converting model to statically quantized version...")
    quantized_model_obj = torch.quantization.convert(model, inplace=True)
    print("Model converted to static quantization.")

    # --- Save the statically quantized model ---
    if not os.path.exists(quantized_model_save_path):
        os.makedirs(quantized_model_save_path)

    # Save the entire quantized model object directly (recommended for torch.quantization)
    torch.save(quantized_model_obj, os.path.join(quantized_model_save_path, "quantized_model_static.pth"))
    print(f"Statically quantized model object saved to: {os.path.join(quantized_model_save_path, 'quantized_model_static.pth')}")

    # Save original config and tokenizer (still needed for AutoTokenizer to load from path)
    model.config.save_pretrained(quantized_model_save_path)
    tokenizer_obj = AutoTokenizer.from_pretrained(model_path)
    tokenizer_obj.save_pretrained(quantized_model_save_path)
    print("Statically quantized model config and tokenizer saved.")
    
    return quantized_model_obj

def prune_PTUP_model(
    model_path: str,
    pruned_model_save_path: str,
    pruning_amount: float = 0.2, # Percentage of weights to prune
    model_name: str = "distilbert-base-uncased", # Needed for AutoModel.from_pretrained if reloading
):
    """
    Applies L1 unstructured pruning to a fine-tuned Question Answering model.
    The pruning is applied post-training and made permanent.

    Args:
        model_path (str): Path to the directory containing the fine-tuned baseline model.
        pruned_model_save_path (str): Path to save the pruned model.
        pruning_amount (float): The percentage of weights to prune (0.0 to 1.0).
        model_name (str): The name of the original pre-trained model (for loading AutoModel correctly).

    Returns:
        transformers.PreTrainedModel: The pruned model object.
    """
    if not os.path.isdir(model_path):
        print(f"Error: Full-precision model not found at '{model_path}'.")
        return None

    print(f"\nLoading model from: {model_path} for pruning...")
    # Load the model. AutoModelForQuestionAnswering will handle .safetensors or .bin
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.eval() # It's good practice to prune models in evaluation mode

    # --- Apply L1 Unstructured Pruning ---
    print(f"Applying L1 unstructured pruning ({pruning_amount*100:.0f}%) to linear layers...")

    # Define the modules (layers) and the attributes (weights) within them to prune.
    # For DistilBERT, we target the weight matrices of its linear layers.
    parameters_to_prune = []
    
    # Prune transformer layers (q_lin, k_lin, v_lin, out_lin, ffn.lin1, ffn.lin2 in each layer)
    for i in range(model.distilbert.config.n_layers): # DistilBERT has 6 transformer layers
        layer = model.distilbert.transformer.layer[i]
        parameters_to_prune.extend([
            (layer.attention.q_lin, 'weight'),
            (layer.attention.k_lin, 'weight'),
            (layer.attention.v_lin, 'weight'),
            (layer.attention.out_lin, 'weight'),
            (layer.ffn.lin1, 'weight'),
            (layer.ffn.lin2, 'weight'),
        ])
    # Prune the final Question Answering head (classifier)
    parameters_to_prune.append((model.qa_outputs, 'weight'))

    # Apply pruning to the specified parameters
    for module, name in parameters_to_prune:
        # prune.l1_unstructured sets the 'name_orig' and 'name_mask' attributes
        # and registers a forward pre-hook.
        prune.l1_unstructured(module, name=name, amount=pruning_amount)

    # --- Remove pruning reparameterization (make it permanent) ---
    # This step is crucial to actually remove the pruned weights and reduce the model size.
    # It removes the masks and makes the pruned weights zero, replacing the original 'weight' attribute.
    print("Removing pruning reparameterization to make pruning permanent...")
    for module, name in parameters_to_prune:
        prune.remove(module, name)
    
    # Note: After prune.remove(), the actual number of non-zero parameters is reduced.
    # Some tools (like count_parameters) might need to be re-run to confirm.

    # --- Save the pruned model ---
    if not os.path.exists(pruned_model_save_path):
        os.makedirs(pruned_model_save_path)

    print(f"Saving pruned model to: {pruned_model_save_path}")
    # Save the pruned model using save_pretrained, which correctly handles config/weights.
    model.save_pretrained(pruned_model_save_path)

    # Copy tokenizer (it's the same as the original fine-tuned model's tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(pruned_model_save_path)
    print("Pruned model and tokenizer saved.")

    return model # Return the pruned model object

def calculate_sparsity(model_path: str) -> float:
    """
    Calculates the sparsity (percentage of zero weights) of a model saved at model_path.
    This function assumes the model is saved after prune.remove().

    Args:
        model_path (str): Path to the directory containing the saved model.

    Returns:
        float: The percentage of zero weights in the model (0.0 to 100.0).
    """
    if not os.path.isdir(model_path):
        print(f"Error: Model directory not found at '{model_path}'.")
        return 0.0

    try:
        # Load the model to inspect its weights.
        # This will load either the full-precision baseline or the pruned model.
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model for sparsity calculation from '{model_path}': {e}")
        print("Ensure the path contains a valid Hugging Face model save with config.json and weights.")
        return 0.0

    total_elements = 0
    zero_elements = 0
    
    # Iterate over all named parameters in the model
    for name, param in model.named_parameters():
        # Focus on weight matrices (typically contain 'weight' in their name)
        # and exclude biases if not pruned, or include all trainable parameters.
        # For L1Unstructured, it typically applies to weight matrices.
        if 'weight' in name and param.dim() > 1: # Exclude 1D biases, focus on layers often pruned
            total_elements += param.numel() # Count total number of elements in the tensor
            zero_elements += torch.sum(param == 0).item() # Count elements exactly equal to 0

    if total_elements == 0:
        sparsity_percentage = 0.0
    else:
        sparsity_percentage = (zero_elements / total_elements) * 100
        
    print(f"Model sparsity at '{model_path}': {sparsity_percentage:.2f}% (zeroed parameters)")
    return sparsity_percentage

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