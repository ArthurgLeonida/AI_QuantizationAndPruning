# src/model_optimizer.py
import torch
import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

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
    # Apply dynamic quantization to the model
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, # Only quantize Linear layers (most common for dynamic)
        dtype=torch.qint8 # Quantize to INT8
    )
    print("Dynamic quantization applied.")

    # Create the directory if it doesn't exist
    if not os.path.exists(quantized_model_save_path):
        os.makedirs(quantized_model_save_path)

    print(f"Saving dynamically quantized model to: {quantized_model_save_path}")
    # Save the quantized model's state_dict
    torch.save(quantized_model.state_dict(), os.path.join(quantized_model_save_path, "pytorch_model.bin"))

    # copy the tokenizer from the original fine-tuned model's path
    # This ensures the quantized model can be loaded with its correct tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(quantized_model_save_path)
    print("Quantized model and tokenizer saved.")
    
    return quantized_model

def measure_model_size(model_path: str):
    """
    Measures the size of a saved model in Megabytes.
    """
    if not os.path.isdir(model_path):
        print(f"Error: Model directory not found at {model_path}.")
        return 0.0
    
    model_file = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(model_file):
        size_bytes = os.path.getsize(model_file)
        size_mb = size_bytes / (1024 * 1024)
        print(f"Model size at '{model_path}': {size_mb:.2f} MB")
        return size_mb
    else:
        print(f"Error: Model file '{model_file}' not found.")
        return 0.0