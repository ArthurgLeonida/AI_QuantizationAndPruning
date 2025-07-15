import multiprocessing
import os
from functools import partial
import torch

# Import modules from src directory
from src.tokenizer_utils import get_tokenizer, prepare_squad_features
from src.model_trainer import train_qa_model # For training the model
from src.metric_utils import compute_squad_metrics # For computing evaluation metrics
from src.model_evaluator import evaluate_fine_tuned_model # For evaluating already fine-tuned model
from datasets import load_dataset, load_from_disk

# Import configuration from config.py
from config import (
    MODEL_NAME,
    MAX_SEQUENCE_LENGTH,
    OVERLAPPING_STRIDE,
    TOKENIZER_SAVE_PATH,
    TOKENIZED_DATASET_SAVE_PATH,
    FINE_TUNED_MODEL_SAVE_PATH, # Path where the trained model will be saved/loaded
    TRAINER_OUTPUT_DIR,
    NUM_TRAIN_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    USE_FP16,
    NUM_PROCESSES_FOR_MAP,
    SUBSET_SIZE, # For testing with smaller dataset
    NO_ANSWER_THRESHOLD
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

    # --- Load Original SQuAD v2 Dataset ---
    print("\nLoading original SQuAD dataset for evaluation examples...")
    original_squad_dataset_dict = load_dataset("squad_v2")
    original_eval_examples = original_squad_dataset_dict["validation"]
    print("Original SQuAD dataset loaded.")

    # --- Handle Tokenizer Loading/Saving ---
    print(f"\nAttempting to load/save tokenizer from/to: {TOKENIZER_SAVE_PATH}")
    parent_tokenizer = get_tokenizer(model_name=MODEL_NAME, save_path=TOKENIZER_SAVE_PATH)
    print("Parent process tokenizer loaded/saved successfully.")

    # --- Step 2: Handle Tokenized Dataset Loading/Saving ---
    # This ensures your tokenized dataset (with labels) is available.
    # It will process it if not found, or load it from disk.
    tokenized_datasets_with_labels = None
    if os.path.isdir(TOKENIZED_DATASET_SAVE_PATH):
        print(f"\nLoading tokenized dataset with labels from local path: {TOKENIZED_DATASET_SAVE_PATH}")
        tokenized_datasets_with_labels = load_from_disk(TOKENIZED_DATASET_SAVE_PATH)
    else:
        print("\nLocal tokenized dataset with labels not found. Preparing SQuAD...")
        print("Original Train Dataset:", original_squad_dataset_dict["train"])
        print("Original Validation Dataset:", original_squad_dataset_dict["validation"])

        print("\nPreparing features and aligning labels for datasets...")
        prepare_features_for_map = partial(
            prepare_squad_features,
            tokenizer_path=TOKENIZER_SAVE_PATH,
            max_length=MAX_SEQUENCE_LENGTH,
            stride=OVERLAPPING_STRIDE
        )
        tokenized_datasets_with_labels = original_squad_dataset_dict.map(
            prepare_features_for_map,
            batched=True,
            num_proc=NUM_PROCESSES_FOR_MAP,
            remove_columns=original_squad_dataset_dict["train"].column_names, # Removes original text columns
            load_from_cache_file=True # Ensures caching is active during map
        )
        print(f"\nSaving tokenized dataset with labels to local path: {TOKENIZED_DATASET_SAVE_PATH}")
        tokenized_datasets_with_labels.save_to_disk(TOKENIZED_DATASET_SAVE_PATH)
        print("Tokenized dataset with labels saved successfully.")

    print("\nFinal Tokenized Datasets with Labels:")
    print(tokenized_datasets_with_labels)
    #print("First tokenized train example with labels:")
    #print(tokenized_datasets_with_labels["train"][0])

    # --- Subsetting the datasets for training/evaluation if SUBSET_SIZE is active ---
    train_dataset_for_trainer = tokenized_datasets_with_labels["train"]
    eval_dataset_for_trainer = tokenized_datasets_with_labels["validation"]
    original_eval_examples_for_metrics = original_eval_examples

    if SUBSET_SIZE > 0 and SUBSET_SIZE < len(tokenized_datasets_with_labels["train"]):
        print(f"\nUsing a SUBSET for training and evaluation.")
        train_dataset_for_trainer = train_dataset_for_trainer.select(range(SUBSET_SIZE))
        
        # For evaluation, use a smaller subset of validation data too
        eval_subset_size = min(SUBSET_SIZE // 10, len(tokenized_datasets_with_labels["validation"]))
        if eval_subset_size == 0 and len(tokenized_datasets_with_labels["validation"]) > 0:
            eval_subset_size = 1 # Ensure at least 1 example if validation data exists
        
        if eval_subset_size > 0:
            eval_dataset_for_trainer = eval_dataset_for_trainer.select(range(eval_subset_size))
            # Filter original evaluation examples by ID to ensure perfect match with tokenized subset
            subset_example_ids = set(eval_dataset_for_trainer["example_id"])
            original_eval_examples_for_metrics = original_eval_examples_for_metrics.filter(
                lambda example: example["id"] in subset_example_ids
            )
            print(f"Subset sizes: Train={len(train_dataset_for_trainer)}, Eval Features={len(eval_dataset_for_trainer)}, Eval Examples={len(original_eval_examples_for_metrics)}")
        else:
            print("Warning: Evaluation subset size is 0 after subsetting.")
    else:
        print("\nUsing full dataset for training and evaluation.")


    # --- Create a partial function for compute_metrics ---
    bound_compute_metrics_for_trainer = partial(
        compute_squad_metrics, # Base function from src/metrics_utils.py
        original_examples=original_eval_examples_for_metrics, # Use (potentially subsetted) original examples
        tokenized_features=eval_dataset_for_trainer,          # Use (potentially subsetted) tokenized features
        tokenizer=parent_tokenizer                            # Use the loaded tokenizer
    )

    '''
    # --- Step 3: Model Training (Uncommented and Enabled) ---
    print("\n--- Starting Model Training Phase ---")
    train_qa_model(
        model_name=MODEL_NAME,
        train_dataset=train_dataset_for_trainer,
        eval_dataset=eval_dataset_for_trainer,
        original_eval_examples=original_eval_examples_for_metrics, # Pass original eval examples
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
        max_train_steps=-1, # Set to a positive integer (e.g., 1000) to train for fixed steps for testing
        no_answer_threshold=NO_ANSWER_THRESHOLD
    )
    print("\n--- Model Training Complete ---")
    '''

    # --- Evaluate the Fine-Tuned Model ---
    print("\n--- Starting Separate Evaluation of Fine-Tuned Model ---")
    evaluate_fine_tuned_model(
        model_path=FINE_TUNED_MODEL_SAVE_PATH,
        tokenizer_path=FINE_TUNED_MODEL_SAVE_PATH, # Tokenizer is saved with the model
        eval_dataset=eval_dataset_for_trainer, # Use the potentially subsetted tokenized eval set
        original_eval_examples=original_eval_examples_for_metrics, # Use the potentially subsetted original eval examples
        compute_metrics_fn=bound_compute_metrics_for_trainer, # Pass the same bound partial function
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        fp16=USE_FP16,
        output_dir="./final_eval_results", # Separate directory for final evaluation logs
        no_answer_threshold=NO_ANSWER_THRESHOLD
    )
    print("\n--- Separate Evaluation Complete ---")