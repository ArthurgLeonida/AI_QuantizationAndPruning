import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
import torch
from functools import partial # Needed for binding arguments to compute_metrics

def evaluate_fine_tuned_model(
    model_path: str,
    tokenizer_path: str,
    tokenized_dataset_path: str,
    original_eval_examples, # Original examples needed for SQuAD metrics
    compute_metrics_fn,
    per_device_eval_batch_size: int = 16,
    fp16: bool = False,
    output_dir: str = "./eval_results", # Directory for evaluation logs
):
    """
    Loads a fine-tuned Question Answering model and evaluates its performance.

    Args:
        model_path (str): Path to the directory containing the saved fine-tuned model.
        tokenizer_path (str): Path to the directory containing the saved tokenizer.
        tokenized_dataset_path (str): Path to the directory containing the tokenized dataset.
        original_eval_examples (datasets.Dataset): The original (untokenized) evaluation examples.
        compute_metrics_fn (callable): The function to compute evaluation metrics (F1, EM).
        per_device_eval_batch_size (int): Batch size for evaluation.
        fp16 (bool): Whether the model was trained with FP16 and should be evaluated with it.
        output_dir (str): Directory for storing evaluation logs.
    """
    if not os.path.isdir(model_path) or not os.path.isdir(tokenizer_path):
        print(f"Error: Model or tokenizer not found at {model_path} or {tokenizer_path}")
        print("Please ensure the model has been fine-tuned and saved correctly.")
        return

    print(f"\nLoading fine-tuned model from: {model_path}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    print("Model loaded successfully.")

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("Tokenizer loaded successfully.")

    print(f"Loading tokenized dataset from: {tokenized_dataset_path}")
    tokenized_datasets = load_from_disk(tokenized_dataset_path)
    eval_dataset = tokenized_datasets["validation"]
    print("Tokenized dataset loaded successfully.")

    # We need TrainingArguments even for evaluation to configure the Trainer
    print("\nSetting up Evaluation Arguments...")
    eval_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=fp16,
        report_to="tensorboard", # Still useful for logging eval results
        # No training-specific args like num_train_epochs, learning_rate etc.
    )

    # The Trainer needs to be initialized for evaluation
    print("\nInitializing Trainer for evaluation...")
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_dataset,
        processing_class=tokenizer, # Use processing_class as per latest transformers
        # Bind original_eval_examples and tokenized_features (eval_dataset) to compute_metrics
        compute_metrics=partial(
            compute_metrics_fn,
            original_examples=original_eval_examples,
            tokenized_features=eval_dataset, # This is your tokenized validation set
            tokenizer=tokenizer # Pass the tokenizer here
        ),
    )
    print("Trainer initialized for evaluation.")

    print("\nStarting model evaluation...")
    metrics = trainer.evaluate()
    print("\nEvaluation Metrics:")
    print(metrics)

    return metrics