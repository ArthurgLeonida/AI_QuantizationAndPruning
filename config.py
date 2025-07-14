# Model and Tokenizer Settings
MODEL_NAME = "distilbert-base-uncased"
MAX_SEQUENCE_LENGTH = 512 # Max length for tokenization
OVERLAPPING_STRIDE = 128 # Stride for overflowing tokens in SQuAD

# Data Paths
TOKENIZER_SAVE_PATH = "./distilbert_tokenizer_local"
TOKENIZED_DATASET_SAVE_PATH = "./squad_tokenized_dataset"
FINE_TUNED_MODEL_SAVE_PATH = "./fine_tuned_baseline_model"
TRAINER_OUTPUT_DIR = "./results"

# Training Hyperparameters
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
USE_FP16 = True # Set to True if your GPU supports it (RTX 3050 does)

# Multiprocessing
NUM_PROCESSES_FOR_MAP = 6 