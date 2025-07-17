# src/model_trainer.py
import os
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from functools import partial # Keep this import

def train_qa_model(
    model_name: str,
    train_dataset,
    eval_dataset,
    original_eval_examples, # Original examples are needed for metrics, passed from main.py
    tokenizer,
    compute_metrics_fn, # This is the base function from metrics_utils.py (compute_squad_metrics)
    output_dir: str = "./results",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    fp16: bool = False,
    save_path: str = "./fine_tuned_baseline_model",
    max_train_steps: int = -1,
    no_answer_threshold: float = 0.0,
):
    """
    Loads, fine-times, and evaluates a Question Answering Transformer model.

    Args:
        model_name (str): The name of the pre-trained model (e.g., "distilbert-base-uncased").
        train_dataset: The tokenized training dataset.
        eval_dataset: The tokenized evaluation dataset.
        original_eval_examples (datasets.Dataset): The original (untokenized) evaluation examples.
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
        max_train_steps (int): Maximum number of training steps.
    """
    print(f"\nLoading {model_name} model for Question Answering...")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    print("Model loaded successfully.")

    print("\nSetting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=500,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1", 
        greater_is_better=True,     
        push_to_hub=False,
        report_to="tensorboard",
        fp16=fp16, # Use mixed-precision training if enabled (check hardware compatibility)
        max_steps=max_train_steps if max_train_steps != -1 else -1,
    )

    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        # Bind original_eval_examples and tokenized_features to compute_metrics_fn
        compute_metrics=partial(
            compute_metrics_fn,
            original_examples=original_eval_examples,
            tokenized_features=eval_dataset,
            tokenizer=tokenizer,
            no_answer_threshold=no_answer_threshold
        ),
    )
    print("Trainer initialized.")

    print("\nStarting model training...")
    trainer.train()
    print("Model training finished.")

    print(f"\nSaving fine-tuned baseline model to: {save_path}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print("Fine-tuned baseline model and tokenizer saved.")