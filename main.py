import multiprocessing
import os
from functools import partial
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, DataCollatorWithPadding
from evaluate import load # New import for loading metrics
# from src.data_loader import load_squad_dataset
from src.tokenizer_utils import get_tokenizer, prepare_squad_features
from datasets import load_dataset, load_from_disk
import torch

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
    
    print(f"Attempting to load/save tokenizer for parent process from/to: {TOKENIZER_SAVE_PATH}")
    parent_tokenizer = get_tokenizer(model_name=MODEL_NAME, save_path=TOKENIZER_SAVE_PATH)
    print("Parent process tokenizer loaded/saved successfully.")

    tokenized_datasets = None

    if os.path.isdir(TOKENIZED_DATASET_SAVE_PATH):
        print(f"\nLoading tokenized dataset from local path: {TOKENIZED_DATASET_SAVE_PATH}")
        tokenized_datasets_with_labels = load_from_disk(TOKENIZED_DATASET_SAVE_PATH)
    else:
        print("\nLocal tokenized dataset not found. Loading and tokenizing SQuAD...")
        squad_dataset_dict = load_dataset("squad")

        print("Train Dataset (original):", squad_dataset_dict["train"])
        print("Validation Dataset (original):", squad_dataset_dict["validation"])

        print("\nPreparing features and aligning labels for datasets...")
        prepare_features_for_map  = partial(
            prepare_squad_features,
            tokenizer_path=TOKENIZER_SAVE_PATH
        )

        tokenized_datasets_with_labels = squad_dataset_dict.map(
            prepare_features_for_map,
            batched=True,
            num_proc=6,
            remove_columns=squad_dataset_dict["train"].column_names,
            load_from_cache_file=True
        )

        print(f"\nSaving tokenized dataset to local path: {TOKENIZED_DATASET_SAVE_PATH}")
        tokenized_datasets_with_labels.save_to_disk(TOKENIZED_DATASET_SAVE_PATH)
        print("Tokenized dataset saved successfully.")

    print("\nFinal Tokenized Datasets with Labels:", tokenized_datasets_with_labels)
    # print("First tokenized train example with labels:", tokenized_datasets_with_labels["train"][0])

# --- Load DistilBERT Model for QA and Fine-Tune it ---
'''
    print("\nLoading DistilBERT model for Question Answering...")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    print("Model loaded successfully.")

    
    # Data Collator: Dynamically pads sequences within a batch to the longest sequence in that batch.
    data_collator = DataCollatorWithPadding(tokenizer=parent_tokenizer)

    # First, load the SQuAD metric
    squad_metric = load("squad")

    def compute_metrics(p):
        start_logits, end_logits = p.predictions # Model outputs raw logits for start/end
        # p.label_ids would contain the start_positions and end_positions labels

        # --- IMPORTANT: PLACE SQuAD POST-PROCESSING LOGIC HERE ---
        # This is where you would convert `start_logits` and `end_logits` into predicted
        # answer text spans and prepare them in the format expected by `squad_metric.compute()`.
        # You will need access to the original evaluation dataset's features (`id`, `context`, `question`)
        # and the `offset_mapping` from your tokenization step to map back to original text.
        #
        # As discussed, for full implementation, refer to Hugging Face's
        # `postprocess_qa_predictions` utility from their Question Answering examples.

        print("\n--- WARNING: SQuAD compute_metrics requires complex post-processing ---")
        print("Please refer to Hugging Face Question Answering examples for `postprocess_qa_predictions` function.")
        print("This function maps model logits back to text answers and generates predictions/references for the SQuAD metric.")
        print("Returning dummy metrics for now.")

        return {
            "f1": 0.0, # Placeholder for calculated F1 score
            "exact_match": 0.0, # Placeholder for calculated Exact Match score
        }

    # --- Setting up Training Arguments ---
    print("\nSetting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=500,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1", # Use F1 as the metric for best model
        greater_is_better=True,     # F1 is better when higher
        push_to_hub=False,
        report_to="tensorboard",
        fp16=True, # Uncomment this if you have a compatible GPU and want to use mixed-precision training for speed
    )

    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets_with_labels["train"],
        # For evaluation, you'll need the original evaluation dataset as well for post-processing:
        eval_dataset=tokenized_datasets_with_labels["validation"],
        # eval_examples=squad_dataset_dict["validation"], # You might need to pass original examples for post-processing
        processing_class=parent_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print("Trainer initialized.")

    # --- Start Training ---
    print("\nStarting model training...")
    trainer.train()
    print("Model training finished.")

    # --- Evaluate the Model ---
    print("\nEvaluating model performance...")
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)

    # --- Save the fine-tuned baseline model and its tokenizer ---
    fine_tuned_model_save_path = "./fine_tuned_baseline_model"
    print(f"\nSaving fine-tuned baseline model to: {fine_tuned_model_save_path}")
    trainer.save_model(fine_tuned_model_save_path)
    parent_tokenizer.save_pretrained(fine_tuned_model_save_path)
    print("Fine-tuned baseline model and tokenizer saved.")
'''