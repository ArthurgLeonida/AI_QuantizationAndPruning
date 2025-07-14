# Efficient Question Answering: Evaluating Optimization Techniques on SQuAD with DistilBERT

## 1. Overview

This project focuses on exploring and evaluating various optimization techniques to enhance the efficiency of Question Answering (QA) models, specifically using **DistilBERT** on the **Stanford Question Answering Dataset (SQuAD)**.

Large language models are powerful but computationally intensive. The primary goal of this project is to:
* Implement and apply different model optimization strategies (e.g., quantization, pruning, knowledge distillation – *you can list specific ones here as you implement them*).
* Fine-tune DistilBERT for extractive QA on the SQuAD dataset.
* Quantitatively assess the impact of these optimizations on model performance (accuracy, F1 score) and efficiency (inference speed, model size).

## 2. Optimization Techniques Explored

This section will detail the specific optimization techniques implemented and evaluated within this project.

* **Quantization:**
    * *Description: (e.g., Reducing precision of weights/activations from float32 to int8/int4).*
    * *Implemented Methods: (e.g., Post-Training Quantization, Quantization-Aware Training).*
* **Pruning:**
    * *Description: (e.g., Removing redundant connections or neurons from the network).*
    * *Implemented Methods: (e.g., Magnitude Pruning, Structured Pruning).*
* **Knowledge Distillation:**
    * *Description: (e.g., Training a smaller 'student' model to mimic the behavior of a larger 'teacher' model).*
    * *Implemented Methods: (e.g., Soft-label distillation).*
* *(Add more techniques here as you integrate them)*

## 3. Dataset

The project utilizes the **Stanford Question Answering Dataset (SQuAD)**, a reading comprehension dataset consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text from the corresponding reading passage.

* **SQuAD 1.1:** Focuses solely on extractive answers present in the text.
* **SQuAD 2.0:** Includes unanswerable questions, requiring the model to determine if no answer is present. (*Specify which version you are using or planning to use*).

## 4. Evaluation Strategy

Model performance will be rigorously evaluated using standard Question Answering metrics:

* **Exact Match (EM):** Measures the percentage of predictions that match the ground truth answer exactly.
* **F1 Score:** Measures the overlap between the prediction and the ground truth answer, calculated as a harmonic mean of precision and recall.

Efficiency metrics such as model size (MB) and inference time (ms) will also be recorded and compared across optimized and baseline models.

## 5. Project Structure

* `main.py`: Main script for fine-tuning baseline and optimized models, and running evaluation.
* `src/`:
    * `data_loader.py`: Handles loading the SQuAD dataset.
    * `tokenizer_utils.py`: Contains the `prepare_squad_features` function for robust tokenization and answer span alignment.
* `requirements.txt`: Lists all Python dependencies.
* `.gitignore`: Specifies files/folders to ignore from version control (e.g., large models, tokenized datasets, logs).
* `README.md`: This project overview.

## 6. Setup

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/ArthurgLeonida/AI_QuantizationAndPruning.git](https://github.com/ArthurgLeonida/AI_QuantizationAndPruning.git)
    cd Project_QntzAndPrun
    ```
    (Replace `Project_QntzAndPrun` with your actual folder name if different)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 7. Usage

To run the full pipeline (data preparation, model fine-tuning, and evaluation):

```bash
python main.py
