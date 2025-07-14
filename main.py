import multiprocessing
import os
from functools import partial
import torch

# Import modules from src directory
from src.tokenizer_utils import get_tokenizer, prepare_squad_features
from src.model_trainer import train_qa_model
from src.metric_utils import compute_squad_metrics # Use the renamed function
from datasets import load_dataset, load_from_disk

# Import configuration
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
    NUM_PROCESSES_FOR_MAP
)


MODEL_NAME = "distilbert-base-uncased"
TOKENIZER_SAVE_PATH = "./distilbert_tokenizer_local"
TOKENIZED_DATASET_SAVE_PATH = "./squad_tokenized_dataset"

if __name__ == '__main__':
    if torch.cuda.is_available(): # This will now be True if setup is correct
        print("CUDA (NVIDIA GPU) is available! Using GPU for acceleration.")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA (NVIDIA GPU) is NOT available. Training will run on CPU.")

    multiprocessing.set_start_method('spawn', force=True)
    
    print(f"\nAttempting to load/save tokenizer from/to: {TOKENIZER_SAVE_PATH}")
    parent_tokenizer = get_tokenizer(model_name=MODEL_NAME, save_path=TOKENIZER_SAVE_PATH)
    print("Parent process tokenizer loaded/saved successfully.")

    tokenized_datasets_with_labels = None

    if os.path.isdir(TOKENIZED_DATASET_SAVE_PATH):
        print(f"\nLoading tokenized dataset with labels from local path: {TOKENIZED_DATASET_SAVE_PATH}")
        tokenized_datasets_with_labels = load_from_disk(TOKENIZED_DATASET_SAVE_PATH)
    else:
        print("\nLocal tokenized dataset with labels not found. Loading and preparing SQuAD...")
        squad_dataset_dict = load_dataset("squad")

        print("Original Train Dataset:", squad_dataset_dict["train"])
        print("Original Validation Dataset:", squad_dataset_dict["validation"])

        print("\nPreparing features and aligning labels for datasets...")
        # Use functools.partial to "bake in" the tokenizer_path and max_length for the map function
        prepare_features_for_map = partial(
            prepare_squad_features,
            tokenizer_path=TOKENIZER_SAVE_PATH,
            max_length=MAX_SEQUENCE_LENGTH,
            stride=OVERLAPPING_STRIDE
        )

        tokenized_datasets_with_labels = squad_dataset_dict.map(
            prepare_features_for_map,
            batched=True,
            num_proc=NUM_PROCESSES_FOR_MAP,
            remove_columns=squad_dataset_dict["train"].column_names, # Remove original columns
            load_from_cache_file=True # Ensure caching is active during map
        )

        print(f"\nSaving tokenized dataset with labels to local path: {TOKENIZED_DATASET_SAVE_PATH}")
        tokenized_datasets_with_labels.save_to_disk(TOKENIZED_DATASET_SAVE_PATH)
        print("Tokenized dataset with labels saved successfully.")

    print("\nFinal Tokenized Datasets with Labels:", tokenized_datasets_with_labels)
    # print("First tokenized train example with labels:", tokenized_datasets_with_labels["train"][0])

    print("\n--- Starting Model Training and Evaluation Phase ---")
    train_qa_model(
        model_name=MODEL_NAME,
        train_dataset=tokenized_datasets_with_labels["train"],
        eval_dataset=tokenized_datasets_with_labels["validation"],
        tokenizer=parent_tokenizer, # Pass the tokenizer object
        compute_metrics_fn=compute_squad_metrics, # Pass the compute_metrics function
        output_dir=TRAINER_OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=USE_FP16,
        save_path=FINE_TUNED_MODEL_SAVE_PATH,
    )
    print("\n--- Model Training and Evaluation Complete ---")