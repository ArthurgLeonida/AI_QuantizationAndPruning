# src/model_trainer.py
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, DataCollatorWithPadding
import torch
import os

def train_qa_model(
    model_name: str,
    train_dataset,
    eval_dataset,
    tokenizer,
    compute_metrics_fn,
    output_dir: str = "./results",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    fp16: bool = False, # Set to True if you have a compatible GPU (RTX 3050)
    save_path: str = "./fine_tuned_baseline_model",
):
    """
    Loads, fine-tunes, and evaluates a Question Answering Transformer model.

    Args:
        model_name (str): The name of the pre-trained model (e.g., "distilbert-base-uncased").
        train_dataset: The tokenized training dataset.
        eval_dataset: The tokenized evaluation dataset.
        tokenizer: The tokenizer object.
        compute_metrics_fn: The function to compute evaluation metrics (F1, EM).
        output_dir (str): Directory for storing logs and checkpoints.
        num_train_epochs (int): Total number of training epochs.
        per_device_train_batch_size (int): Batch size for training.
        per_device_eval_batch_size (int): Batch size for evaluation.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): L2 regularization.
        fp16 (bool): Whether to use mixed-precision training (True for RTX cards).
        save_path (str): Directory to save the fine-tuned model and tokenizer.
    """
    print(f"\nLoading {model_name} model for Question Answering...")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    print("Model loaded successfully.")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("\nSetting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch", # Corrected argument name
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=500,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1", # Use F1 as the metric for best model
        greater_is_better=True,     # F1 is better when higher
        push_to_hub=False,
        report_to="tensorboard",
        fp16=fp16, # Use mixed-precision training if enabled
    )

    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer, # Use 'processing_class' instead of 'tokenizer'
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    print("Trainer initialized.")

    print("\nStarting model training...")
    trainer.train()
    print("Model training finished.")

    print("\nEvaluating model performance...")
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)

    print(f"\nSaving fine-tuned baseline model to: {save_path}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path) # Save the tokenizer with the fine-tuned model
    print("Fine-tuned baseline model and tokenizer saved.")