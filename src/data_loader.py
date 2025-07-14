from datasets import load_dataset

def load_squad_dataset():
    """
    Loads the SQuAD dataset and returns the train and validation splits.
    """
    squad_dataset = load_dataset("squad")
    return squad_dataset["train"], squad_dataset["validation"]

# You might add other data loading functions here in the future
# def load_custom_dataset(path):
#     # ...