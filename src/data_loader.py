import os
from functools import partial
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset 
from src.tokenizer_utils import get_tokenizer, prepare_squad_features 

def load_squad_dataset_dict(version="squad"):
    """
    Loads the SQuAD dataset (or squad_v2) as a DatasetDict.

    Args:
        version (str): "squad" for v1.1 or "squad_v2" for v2.0.

    Returns:
        datasets.DatasetDict: The loaded SQuAD dataset.
    """
    print(f"Loading SQuAD dataset version: {version}")
    return load_dataset(version)


def get_subsetted_datasets(
    tokenized_datasets_with_labels: DatasetDict,
    original_eval_examples: Dataset,
    subset_size: int
) -> tuple[Dataset, Dataset, Dataset]: # Returns tuple of (train_ds, eval_ds, original_eval_ds)
    """
    Applies subsetting logic to the tokenized training/validation datasets and
    the original evaluation examples.

    Args:
        tokenized_datasets_with_labels (DatasetDict): The full tokenized dataset (with "train" and "validation" splits).
        original_eval_examples (Dataset): The full original evaluation dataset.
        subset_size (int): The desired size for the training subset.
                            If <= 0 or too large, the full dataset is used.

    Returns:
        tuple: (train_dataset_for_trainer, eval_dataset_for_trainer, original_eval_examples_for_metrics)
               where each element is a datasets.Dataset object, potentially subsetted.
    """
    train_dataset_for_trainer = tokenized_datasets_with_labels["train"]
    eval_dataset_for_trainer = tokenized_datasets_with_labels["validation"]
    original_eval_examples_for_metrics = original_eval_examples

    if subset_size > 0 and subset_size < len(tokenized_datasets_with_labels["train"]):
        print(f"\nUsing a SUBSET for training and evaluation (Train size: {subset_size}).")
        train_dataset_for_trainer = train_dataset_for_trainer.select(range(subset_size))
        
        # For evaluation, use a smaller subset of validation data, proportional to train subset
        eval_subset_size = min(subset_size // 10, len(tokenized_datasets_with_labels["validation"]))
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
    
    return train_dataset_for_trainer, eval_dataset_for_trainer, original_eval_examples_for_metrics

def load_and_prepare_data(
    model_name: str,
    tokenizer_save_path: str,
    tokenized_dataset_save_path: str,
    max_sequence_length: int,
    overlapping_stride: int,
    num_processes_for_map: int
) -> tuple[DatasetDict, Dataset, any, DatasetDict]:
    """
    Consolidates loading of original SQuAD dataset, tokenizer, and tokenized dataset preparation/saving.

    Args:
        model_name (str): Name of the pre-trained model for tokenizer.
        tokenizer_save_path (str): Path to save/load tokenizer.
        tokenized_dataset_save_path (str): Path to save/load tokenized dataset.
        max_sequence_length (int): Max sequence length for tokenization.
        overlapping_stride (int): Stride for overflowing tokens.
        num_processes_for_map (int): Number of processes for datasets.map().

    Returns:
        tuple: (original_squad_dataset_dict, original_eval_examples, parent_tokenizer, tokenized_datasets_with_labels)
    """

    # --- Load Original SQuAD v2 Dataset ---
    print("\nLoading original SQuAD dataset for evaluation examples...")
    original_squad_dataset_dict = load_squad_dataset_dict(version="squad_v2")
    original_eval_examples = original_squad_dataset_dict["validation"]
    print("Original SQuAD dataset loaded.")

    # --- Handle Tokenizer Loading/Saving ---
    print(f"\nAttempting to load/save tokenizer from/to: {tokenizer_save_path}")
    parent_tokenizer = get_tokenizer(model_name=model_name, save_path=tokenizer_save_path)
    print("Parent process tokenizer loaded/saved successfully.")

    # --- Handle Tokenized Dataset Loading/Saving ---
    tokenized_datasets_with_labels = None
    if os.path.isdir(tokenized_dataset_save_path):
        print(f"\nLoading tokenized dataset with labels from local path: {tokenized_dataset_save_path}")
        tokenized_datasets_with_labels = load_from_disk(tokenized_dataset_save_path)
    else:
        print("\nLocal tokenized dataset with labels not found. Preparing SQuAD...")
        print("Original Train Dataset:", original_squad_dataset_dict["train"])
        print("Original Validation Dataset:", original_squad_dataset_dict["validation"])

        print("\nPreparing features and aligning labels for datasets...")
        prepare_features_for_map = partial(
            prepare_squad_features,
            tokenizer_path=tokenizer_save_path,
            max_length=max_sequence_length,
            stride=overlapping_stride
        )
        tokenized_datasets_with_labels = original_squad_dataset_dict.map(
            prepare_features_for_map,
            batched=True,
            num_proc=num_processes_for_map,
            remove_columns=original_squad_dataset_dict["train"].column_names, 
            load_from_cache_file=True 
        )
        print(f"\nSaving tokenized dataset with labels to local path: {tokenized_dataset_save_path}")
        tokenized_datasets_with_labels.save_to_disk(tokenized_dataset_save_path)
        print("Tokenized dataset with labels saved successfully.")

    print("\nFinal Tokenized Datasets with Labels:")
    print(tokenized_datasets_with_labels)
    #print("First tokenized train example with labels:")
    #print(tokenized_datasets_with_labels["train"][0])
    
    return original_squad_dataset_dict, original_eval_examples, parent_tokenizer, tokenized_datasets_with_labels