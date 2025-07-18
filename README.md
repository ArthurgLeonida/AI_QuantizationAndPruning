# Evaluating the Impact of Quantization and Pruning on DistilBERT for Question Answering

This project investigates the application and impact of model optimization techniques, specifically **Post-Training Dynamic Quantization (PTQ)** and **Post-Training Unstructured Pruning (PTUP)**, on a DistilBERT model. The model is fine-tuned for Question Answering (QA) on the SQuAD v2.0 dataset, with a focus on understanding the trade-offs between model performance, size, and inference speed.

## Project Paper

You can find the detailed paper discussing the methodology, experiments, and results of this project here:

[**Download the Paper (PDF)**](https://github.com/ArthurgLeonida/AI_QuantizationAndPruning/releases/download/v2.0-final/QuantizationAndPruning_Paper.pdf).

## Features

* **Efficient Data Pipeline:** Leverages Hugging Face's `datasets` library for robust and efficient data loading, tokenization, label alignment, multiprocessing, and local caching of processed data.
* **Modular Code Structure:** Project is organized into distinct modules (`src/`, `config.py`) for clarity, maintainability, and reusability.
* **Baseline Model Training & Evaluation:** Fine-tunes a DistilBERT baseline model on SQuAD v2.0 using `transformers.Trainer`, providing comprehensive evaluation metrics (F1, EM, HasAns/NoAns breakdown).
* **Post-Training Dynamic Quantization (PTQ):** Implements dynamic quantization to reduce model size and observe its impact on performance and inference speed.
* **Post-Training Unstructured Pruning (PTUP):** Applies unstructured pruning to introduce sparsity in the model and assess its effects on performance, size, and speed.
* **Automated Benchmarking:** Includes dedicated functions for precise measurement of model size and inference speed.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ArthurgLeonida/AI_QuantizationAndPruning.git](https://github.com/ArthurgLeonida/AI_QuantizationAndPruning.git)
    cd AI_QuantizationAndPruning
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you install the CUDA-enabled PyTorch version compatible with your GPU for optimal performance, as per instructions at PyTorch.org.)*

4.  **Configure `config.py`:**
    Review and adjust hyperparameters and paths in `config.py` as needed (e.g., `NUM_TRAIN_EPOCHS`, `SUBSET_SIZE`, `PRUNING_AMOUNT`).

## Usage

To run the full project pipeline (data preparation, baseline training, and optimization evaluations):

```bash
python main.py