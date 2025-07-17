import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments, AutoConfig
import torch
from functools import partial

def evaluate_fine_tuned_model(
    model_path: str,
    tokenizer_path: str,
    eval_dataset,
    original_eval_examples,
    compute_metrics_fn,
    per_device_eval_batch_size: int = 16,
    fp16: bool = False,
    output_dir: str = "./eval_results",
    no_answer_threshold: float = 0.0,
    is_quantized: bool = False,
):
    """
    Loads a fine-tuned or quantized Question Answering model and evaluates its performance.
    """
    if not os.path.isdir(model_path) or not os.path.isdir(tokenizer_path):
        print(f"Error: Model or tokenizer directory not found at {model_path} or {tokenizer_path}")
        print("Please ensure the model has been fine-tuned and saved correctly.")
        return None

    print(f"\nLoading model from: {model_path}")
    
    if is_quantized:
        # Load the entire quantized model object directly from the .pth file.
        try:
            model = torch.load(os.path.join(model_path, "quantized_model.pth"), map_location=torch.device("cpu"), weights_only=False)
            model.eval()
            model.to(torch.device("cpu")) # dynamic quantization is CPU-optimized
            print("Quantized model object loaded successfully.")
        except Exception as e:
            print(f"Error loading quantized model object from {os.path.join(model_path, 'quantized_model.pth')}: {e}")
            print("Ensure it was saved as a full object via torch.save(model_obj, path).")
            return None 

    else:
        # --- Loading for Full-Precision (Non-Quantized) Models ---
        print("Loading full-precision model...")
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        model.eval()
        print("Full-precision model loaded successfully.")

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("Tokenizer loaded successfully.")

    print("\nSetting up Evaluation Arguments...")
    eval_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=fp16,
        report_to="tensorboard",
        no_cuda=is_quantized,
    )

    print("\nInitializing Trainer for evaluation...")
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=partial(
            compute_metrics_fn,
            original_examples=original_eval_examples,
            tokenized_features=eval_dataset,
            tokenizer=tokenizer,
            no_answer_threshold=no_answer_threshold
        ),
    )
    print("Trainer initialized for evaluation.")

    print("\nStarting model evaluation...")
    metrics = trainer.evaluate()
    print("\nEvaluation Metrics:")
    print(metrics)

    return metrics