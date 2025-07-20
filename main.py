import multiprocessing
import os
from functools import partial
import torch
# Import modules from src directory
from src.model_trainer import train_qa_model 
from src.metric_utils import compute_squad_metrics 
from src.model_evaluator import evaluate_fine_tuned_model 
from src.model_optmizer import quantize_PTQ_model, measure_model_size, benchmark_inference_speed, prune_PTUP_model, calculate_sparsity
from src.data_loader import get_subsetted_datasets, load_and_prepare_data
# Import configuration from config.py
from config import (
    MODEL_NAME,
    MAX_SEQUENCE_LENGTH,
    OVERLAPPING_STRIDE,
    TOKENIZER_SAVE_PATH,
    TOKENIZED_DATASET_SAVE_PATH,
    FINE_TUNED_MODEL_SAVE_PATH, 
    TRAINER_OUTPUT_DIR,
    NUM_TRAIN_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    USE_FP16,
    NUM_PROCESSES_FOR_MAP,
    SUBSET_SIZE, 
    NO_ANSWER_THRESHOLD,
    QUANTIZED_MODEL_SAVE_PATH,
    PRUNED_MODEL_SAVE_PATH,
    PRUNING_AMOUNT
)

if __name__ == '__main__':
    # Initial setup for multiprocessing (crucial for Windows and PyTorch/CUDA)
    multiprocessing.set_start_method('spawn', force=True)

    # --- Verify GPU/Device ---
    if torch.cuda.is_available():
        print("CUDA (NVIDIA GPU) is available! Using GPU for acceleration.")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA (NVIDIA GPU) is NOT available. Training will run on CPU.")

    # --- Load and Prepare Data ---
    original_squad_dataset_dict, original_eval_examples, parent_tokenizer, tokenized_datasets_with_labels = load_and_prepare_data(
            model_name=MODEL_NAME,
            tokenizer_save_path=TOKENIZER_SAVE_PATH,
            tokenized_dataset_save_path=TOKENIZED_DATASET_SAVE_PATH,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            overlapping_stride=OVERLAPPING_STRIDE,
            num_processes_for_map=NUM_PROCESSES_FOR_MAP
        )

    # --- Subsetting the datasets for training/evaluation if SUBSET_SIZE is active ---
    train_dataset_for_trainer, eval_dataset_for_trainer, original_eval_examples_for_metrics = get_subsetted_datasets(
            tokenized_datasets_with_labels=tokenized_datasets_with_labels,
            original_eval_examples=original_eval_examples,
            subset_size=SUBSET_SIZE
        )

    # --- Create a partial function for compute_metrics ---
    bound_compute_metrics_for_trainer = partial(
        compute_squad_metrics, 
        original_examples=original_eval_examples_for_metrics, # original examples
        tokenized_features=eval_dataset_for_trainer,          # tokenized features
        tokenizer=parent_tokenizer                            # loaded tokenizer
    )

    ################################## Fine-Tuned Baseline Model Training ##################################
    
    print("\n--- Starting Model Training Phase ---")
    train_qa_model(
        model_name=MODEL_NAME,
        train_dataset=train_dataset_for_trainer,
        eval_dataset=eval_dataset_for_trainer,
        original_eval_examples=original_eval_examples_for_metrics, 
        tokenizer=parent_tokenizer,
        compute_metrics_fn=bound_compute_metrics_for_trainer,
        output_dir=TRAINER_OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=USE_FP16,
        save_path=FINE_TUNED_MODEL_SAVE_PATH,
        max_train_steps=-1, 
        no_answer_threshold=NO_ANSWER_THRESHOLD
    )
    print("\n--- Model Training Complete ---")
    
    print("\n--- Baseline Model Size ---")
    measure_model_size(FINE_TUNED_MODEL_SAVE_PATH)

    print("\n--- Baseline Model Sparsity ---")
    baseline_sparsity = calculate_sparsity(FINE_TUNED_MODEL_SAVE_PATH)
    print(f"Baseline model has {baseline_sparsity:.2f}%.")

    print("\n--- Benchmarking Baseline Model Inference Speed ---")
    baseline_gpu_samples_per_sec = benchmark_inference_speed(
        model_path=FINE_TUNED_MODEL_SAVE_PATH,
        tokenizer_path=TOKENIZER_SAVE_PATH,
        is_quantized=False,
        device_str="cpu", # CPU/GPU inference speed
        batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        sequence_length=MAX_SEQUENCE_LENGTH 
    )
    print(f"Baseline (FP32) Samples/Sec (GPU): {baseline_gpu_samples_per_sec:.2f}")
    
    evaluate_fine_tuned_model(
        model_path=FINE_TUNED_MODEL_SAVE_PATH,
        tokenizer_path=FINE_TUNED_MODEL_SAVE_PATH, 
        eval_dataset=eval_dataset_for_trainer, 
        original_eval_examples=original_eval_examples_for_metrics, 
        compute_metrics_fn=bound_compute_metrics_for_trainer, 
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        fp16=USE_FP16,
        is_quantized = False,
        output_dir="./final_eval_results", 
        no_answer_threshold=NO_ANSWER_THRESHOLD
    )
    print("\n--- Evaluation Complete ---")
    
    ################################## Post-Training Dynamic Quantization ##################################

    print("\n--- Starting Post-Training Quantization and Evaluation ---")
    quantized_model = quantize_PTQ_model(model_path=FINE_TUNED_MODEL_SAVE_PATH, quantized_model_save_path=QUANTIZED_MODEL_SAVE_PATH)
    
    if quantized_model: 
        print("\n--- Quantized Model Size ---")
        quantized_model_file_size_mb = os.path.getsize(os.path.join(QUANTIZED_MODEL_SAVE_PATH, "quantized_model.pth")) / (1024 * 1024)
        print(f"Model size at '{QUANTIZED_MODEL_SAVE_PATH}' (quantized_model.pth): {quantized_model_file_size_mb:.2f} MB")

        print("\n--- Benchmarking Quantized Model Inference Speed ---")
        quantized_cpu_samples_per_sec = benchmark_inference_speed(
            model_path=QUANTIZED_MODEL_SAVE_PATH,
            tokenizer_path=QUANTIZED_MODEL_SAVE_PATH,
            is_quantized=True,
            device_str="cpu", # Quantized models (Dynamic) must run on CPU
            batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            sequence_length=MAX_SEQUENCE_LENGTH 
        )
        print(f"Quantized (INT8) Samples/Sec (CPU): {quantized_cpu_samples_per_sec:.2f}")

        print("\n--- Quantized Model Evaluation ---")
        evaluate_fine_tuned_model(
            model_path=QUANTIZED_MODEL_SAVE_PATH,
            tokenizer_path=QUANTIZED_MODEL_SAVE_PATH,
            eval_dataset=eval_dataset_for_trainer,
            original_eval_examples=original_eval_examples_for_metrics,
            compute_metrics_fn=bound_compute_metrics_for_trainer,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            fp16=False, # INT8 quantized models typically run on CPU or specialized hardware, not FP16
            is_quantized = True,
            output_dir=os.path.join(TRAINER_OUTPUT_DIR, "quantized_eval_results"), 
            no_answer_threshold=NO_ANSWER_THRESHOLD
        )
    print("\n--- Post-Training Quantization Complete ---")

    ################################## Post-Training Unstructured Pruning ##################################

    print("\n--- Starting Pruning and Evaluation ---")
    pruned_model = prune_PTUP_model(
        model_path=FINE_TUNED_MODEL_SAVE_PATH,
        pruned_model_save_path=PRUNED_MODEL_SAVE_PATH,
        pruning_amount=PRUNING_AMOUNT
    )

    if pruned_model:
        print("\n--- Pruned Model Size ---")
        measure_model_size(PRUNED_MODEL_SAVE_PATH)

        print("\n--- Pruned Model Sparsity ---")
        pruned_sparsity = calculate_sparsity(PRUNED_MODEL_SAVE_PATH)
        print(f"Pruned model has {pruned_sparsity:.2f}% sparsity.")

        print("\n--- Benchmarking Pruned Model Inference Speed ---")
        pruned_gpu_samples_per_sec = benchmark_inference_speed(
            model_path=PRUNED_MODEL_SAVE_PATH,
            tokenizer_path=PRUNED_MODEL_SAVE_PATH,
            is_quantized=False,
            device_str="cuda" if torch.cuda.is_available() else "cpu", # Pruned model can still run on GPU
            batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            sequence_length=MAX_SEQUENCE_LENGTH
        )
        print(f"Pruned Samples/Sec ({'GPU' if torch.cuda.is_available() else 'CPU'}): {pruned_gpu_samples_per_sec:.2f}")

        print("\n--- Pruned Model Evaluation ---")
        evaluate_fine_tuned_model(
            model_path=PRUNED_MODEL_SAVE_PATH,
            tokenizer_path=PRUNED_MODEL_SAVE_PATH,
            eval_dataset=eval_dataset_for_trainer,
            original_eval_examples=original_eval_examples_for_metrics,
            compute_metrics_fn=bound_compute_metrics_for_trainer,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            fp16=USE_FP16, # Pruned model should still use FP16 if baseline did
            is_quantized=False,
            output_dir=os.path.join(TRAINER_OUTPUT_DIR, "pruned_eval_results"),
            no_answer_threshold=NO_ANSWER_THRESHOLD
        )
    print("\n--- Post Training Unstructured Pruning Complete ---")